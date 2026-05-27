import copy
import importlib
import logging
import math
import os
import shutil
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)

# from diffusers.training_utils import compute_snr # diffusers is still working on this, uncomment in future versions
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo
from packaging import version
from PIL import Image
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from src.data.dataset import GlobDataset, MoviPairDataset
from src.metrics.segmentation import (
    per_image_fg_ari,
    per_image_mbo,
    per_image_miou,
)
from src.models.encoder import (
    LATENT_ENCODER_CLASSES,
    DinoSlotAttentionEncoder,
    SlotAttentionEncoder,
    build_latent_encoder,
)
from src.parser import parse_args
from src.pipeline.composable_stable_diffusion_pipeline import (
    ComposableStableDiffusionPipeline,
)


if is_wandb_available():
    import wandb

logger = get_logger(__name__)

SLOT_ATTN_ENCODER_CLASSES = (SlotAttentionEncoder, DinoSlotAttentionEncoder)


def compose_eps(unet, noisy_model_input, timesteps, slot_tokens, K, R):
    """Compose an epsilon prediction by summing per-slot epsilons.

    Each per-slot forward conditions on ``[slot_k, registers]``; the composed
    prediction is ``sum_k eps_slot_k``. Registers (if any) ride along in every
    slot's conditioning sequence but there is no separate registers-only
    forward.
    """
    B = noisy_model_input.shape[0]
    slots = slot_tokens[:, :K]
    registers = slot_tokens[:, K:] if R > 0 else None

    slot_cond = slots.unsqueeze(2)  # [B, K, 1, D]
    if R > 0:
        regs_exp = registers.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, R, D]
        slot_cond = torch.cat([slot_cond, regs_exp], dim=2)  # [B, K, 1+R, D]
    cond = slot_cond.flatten(0, 1)  # [B*K, 1+R, D]

    noisy_expanded = noisy_model_input[:, None].expand(-1, K, -1, -1, -1).flatten(0, 1)
    timesteps_expanded = timesteps[:, None].expand(-1, K).flatten(0, 1)

    eps_slots = unet(noisy_expanded, timesteps_expanded, cond).sample
    eps_slots = eps_slots.view(B, K, *eps_slots.shape[1:])  # [B, K, C, H, W]
    return eps_slots.sum(dim=1)


@torch.no_grad()
def log_validation(
    val_dataset,
    latent_encoder,
    unet,
    vae,
    scheduler,
    args,
    accelerator,
    weight_dtype,
    global_step,
    movi_eval_dataset=None,
):
    logger.info("Running validation... \n.")
    unet = accelerator.unwrap_model(unet)
    latent_encoder = accelerator.unwrap_model(latent_encoder)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    # Hold on to the training (DDPM) scheduler for the val-loss computation
    # below; only the image-generation pipeline swaps to a faster inference
    # scheduler.
    noise_scheduler = scheduler

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in scheduler.config:
        variance_type = scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    # use a more efficient scheduler at test time
    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    scheduler = scheduler_class.from_config(scheduler.config, **scheduler_args)

    pipeline = ComposableStableDiffusionPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=None,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        None
        if args.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )

    num_digits = len(str(args.max_train_steps))
    folder_name = f"image_logging_{global_step:0{num_digits}}"
    image_log_dir = os.path.join(
        accelerator.logging_dir,
        folder_name,
    )
    os.makedirs(image_log_dir, exist_ok=True)

    images = []
    image_count = 0

    for batch_idx, batch in enumerate(val_dataloader):
        pixel_values = batch["pixel_values"].to(
            device=accelerator.device, dtype=weight_dtype
        )

        with torch.autocast("cuda"):
            model_input = vae.encode(pixel_values).latent_dist.sample()
            pixel_values_recon = vae.decode(model_input).sample

            slot_tokens = latent_encoder(pixel_values)  # [B, K+R, D]
            K = latent_encoder.num_components
            R = getattr(latent_encoder, "num_registers", 0)
            slots = slot_tokens[:, :K]
            registers = slot_tokens[:, K:] if R > 0 else None

            # one generation per slot, then the full reconstruction from all slots
            per_slot_embeds = [
                slots[:, s : s + 1, :]
                if registers is None
                else torch.cat([slots[:, s : s + 1, :], registers], dim=1)
                for s in range(K)
            ]
            # Per-slot decode: scale by K so the noise prediction has the
            # same magnitude as the full sum_k eps_slot_k. Without this, a
            # single slot contributes ~noise/K and the DDIM step barely
            # denoises -- the per-slot image stays near the init noise.
            per_slot_images = [
                pipeline(
                    prompt_embeds=embeds.to(
                        device=accelerator.device, dtype=weight_dtype
                    ),
                    num_registers=R,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=25,
                    generator=generator,
                    guidance_scale=float(K),
                    output_type="pt",
                ).images
                for embeds in per_slot_embeds
            ]

            images_recon = pipeline(
                prompt_embeds=slot_tokens,
                num_registers=R,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=25,
                generator=generator,
                guidance_scale=1.0,
                output_type="pt",
            ).images

        grid_image = torch.cat(
            [pixel_values.unsqueeze(1) * 0.5 + 0.5]
            + [img.unsqueeze(1) for img in per_slot_images]
            + [images_recon.unsqueeze(1)],
            dim=1,
        )
        grid_image = make_grid(
            grid_image.view(
                grid_image.shape[0] * grid_image.shape[1],
                grid_image.shape[2],
                grid_image.shape[3],
                grid_image.shape[4],
            ),
            nrow=grid_image.shape[1],
        )
        ndarr = (
            grid_image.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        im = Image.fromarray(ndarr)
        images.append(im)
        img_path = os.path.join(image_log_dir, f"image_{batch_idx:02}.jpg")
        im.save(img_path, optimize=True, quality=95)
        image_count += pixel_values.shape[0]
        if image_count >= args.num_validation_images:
            break

    # --- Scalar validation metrics ------------------------------------------
    # Cheap single-step diffusion loss on val batches, using the same
    # compose_eps composition as training. Also logs the mean off-diagonal
    # pairwise cosine similarity of object slots -- a diagnostic for slot
    # collapse (high values => slots are redundant).
    K = latent_encoder.num_components
    R = getattr(latent_encoder, "num_registers", 0)
    metric_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    num_metric_batches = max(1, args.num_validation_images // args.val_batch_size)
    metric_seed = args.seed if args.seed is not None else 0
    metric_generator = torch.Generator(device=accelerator.device).manual_seed(
        metric_seed + global_step
    )
    val_losses = []
    slot_cos_sims = []
    for i, batch in enumerate(metric_loader):
        if i >= num_metric_batches:
            break
        pixel_values = batch["pixel_values"].to(
            device=accelerator.device, dtype=weight_dtype
        )
        with torch.autocast("cuda"):
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor
            noise = torch.randn(
                model_input.shape,
                generator=metric_generator,
                device=accelerator.device,
                dtype=model_input.dtype,
            )
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (pixel_values.shape[0],),
                generator=metric_generator,
                device=accelerator.device,
            ).long()
            noisy = noise_scheduler.add_noise(model_input, noise, timesteps)

            slot_tokens = latent_encoder(pixel_values)
            model_pred = compose_eps(unet, noisy, timesteps, slot_tokens, K, R)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )
            val_losses.append(F.mse_loss(model_pred.float(), target.float()).item())

            slots = slot_tokens[:, :K].float()
            slots_n = F.normalize(slots, dim=-1)
            sim = torch.einsum("bkd,bjd->bkj", slots_n, slots_n)
            eye = torch.eye(K, device=slots.device, dtype=torch.bool)
            slot_cos_sims.append(sim[:, ~eye].mean().item())

    val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
    slot_cos = (
        sum(slot_cos_sims) / len(slot_cos_sims) if slot_cos_sims else float("nan")
    )

    # --- Object-discovery metrics (MOVi-E only, when GT seg + slot-attn encoder)
    fg_ari = mbo = miou = None
    if movi_eval_dataset is not None and isinstance(
        latent_encoder, SLOT_ATTN_ENCODER_CLASSES
    ):
        seg_loader = torch.utils.data.DataLoader(
            movi_eval_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
        )
        aris, mbos, mious = [], [], []
        for batch in seg_loader:
            pixel_values = batch["pixel_values"].to(
                device=accelerator.device, dtype=weight_dtype
            )
            gt = batch["segment"].to(device=accelerator.device)
            _, attn = latent_encoder(pixel_values, return_attn=True)
            attn_up = F.interpolate(
                attn.float(),
                size=(args.resolution, args.resolution),
                mode="bilinear",
                align_corners=False,
            )
            pred_mask = attn_up.argmax(dim=1)
            for b in range(pred_mask.shape[0]):
                gt_np = gt[b].cpu().numpy()
                pred_np = pred_mask[b].cpu().numpy()
                a = per_image_fg_ari(gt_np, pred_np)
                m = per_image_mbo(gt_np, pred_np)
                u = per_image_miou(gt_np, pred_np, ignore_background=False)
                if a is not None:
                    aris.append(a)
                if m is not None:
                    mbos.append(m)
                if u is not None:
                    mious.append(u)
        fg_ari = float(np.mean(aris)) if aris else float("nan")
        mbo = float(np.mean(mbos)) if mbos else float("nan")
        miou = float(np.mean(mious)) if mious else float("nan")

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, global_step, dataformats="NHWC"
            )
            tracker.writer.add_scalar("val_loss", val_loss, global_step)
            tracker.writer.add_scalar("slot_pairwise_cos", slot_cos, global_step)
            if fg_ari is not None:
                tracker.writer.add_scalar("val/fg_ari", fg_ari, global_step)
                tracker.writer.add_scalar("val/mbo", mbo, global_step)
                tracker.writer.add_scalar("val/miou", miou, global_step)
        if tracker.name == "wandb":
            # Step is implicit -- wandb merges into the current internal step,
            # which tracks `global_step` via the per-step accelerator.log()
            # call in the training loop.
            wandb_log = {
                "validation": [
                    wandb.Image(image, caption=f"{i}") for i, image in enumerate(images)
                ],
                "val_loss": val_loss,
                "slot_pairwise_cos": slot_cos,
            }
            if fg_ari is not None:
                wandb_log["val/fg_ari"] = fg_ari
                wandb_log["val/mbo"] = mbo
                wandb_log["val/miou"] = miou
            tracker.log(wandb_log)
    torch.cuda.empty_cache()

    return images


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load scheduler and models
    if args.unet_config == "pretrain_sd":
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name, subfolder="scheduler"
        )
    else:
        noise_scheduler_config = DDPMScheduler.load_config(args.scheduler_config)
        noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name, subfolder="vae")

    # The encoder class (LatentEncoder baseline vs SlotAttentionEncoder) is
    # selected by the `_class_name` field of the latent-encoder config json.
    latent_encoder = build_latent_encoder(args.latent_encoder_config)

    if os.path.exists(args.unet_config):
        train_unet = True
        unet_config = UNet2DConditionModel.load_config(args.unet_config)
        unet = UNet2DConditionModel.from_config(unet_config)
    elif args.unet_config == "pretrain_sd":
        train_unet = False
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name, subfolder="unet", revision=args.revision
        )
    else:
        raise ValueError(f"Unknown unet config {args.unet_config}")

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                # continue if not a latent encoder or the UNet
                if not isinstance(
                    model, LATENT_ENCODER_CLASSES + (UNet2DConditionModel,)
                ):
                    continue

                sub_dir = model._get_name().lower()

                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            sub_dir = model._get_name().lower()

            if isinstance(model, LATENT_ENCODER_CLASSES):
                load_model = type(model).from_pretrained(input_dir, subfolder=sub_dir)
                model.register_to_config(**load_model.config)
            elif isinstance(model, UNet2DConditionModel):
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder=sub_dir
                )
                model.register_to_config(**load_model.config)
            else:
                raise ValueError(f"Unknown model type {type(model)}")

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    if not train_unet:
        unet.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if train_unet and accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if accelerator.unwrap_model(latent_encoder).dtype != torch.float32:
        raise ValueError(
            f"Slot Attn loaded as datatype {accelerator.unwrap_model(latent_encoder).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = list(latent_encoder.parameters()) + (
        list(unet.parameters()) if train_unet else []
    )
    params_group = [
        {
            "params": list(latent_encoder.parameters()),
            "lr": args.learning_rate * args.encoder_lr_scale,
        }
    ]
    if train_unet:
        params_group.append({"params": unet.parameters(), "lr": args.learning_rate})

    optimizer = optimizer_class(
        params_group,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    """def warm_and_decay_lr_scheduler(step: int):
        warmup_steps = 10000
        decay_steps = 100000
        if step < warmup_steps:
            factor = step / warmup_steps
        else:
            factor = 1
        #factor *= 0.5 ** (step / decay_steps)
        return factor
    
    # implement your lr_sceduler here, here I use constant functions as 
    # the template for your reference
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warm_and_decay_lr_scheduler
        )"""

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda _: 1, lambda _: 1] if train_unet else [lambda _: 1]
    )

    train_dataset = GlobDataset(
        root=args.dataset_root,
        img_size=args.resolution,
        img_glob=args.dataset_glob,
        data_portion=(0.0, args.train_split_portion),
        random_flip=args.flip_images,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # validation set is only for visualization
    val_dataset = GlobDataset(
        root=args.dataset_root,
        img_size=args.resolution,
        img_glob=args.dataset_glob,
        data_portion=(
            args.train_split_portion if args.train_split_portion < 1.0 else 0.9,
            1.0,
        ),
    )

    # Optional MOVi-E (image, GT segment) dataset for training-time
    # object-discovery metrics. Only built when the flag is set; the segmentation
    # eval inside log_validation also requires a slot-attention encoder.
    movi_eval_dataset = None
    if args.movi_eval_root is not None:
        movi_eval_dataset = MoviPairDataset(
            root=args.movi_eval_root,
            split=args.movi_eval_split,
            img_size=args.resolution,
            max_images=args.movi_eval_max_images,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Cache slot / register counts before DDP-wrapping the encoder -- the
    # DistributedDataParallel wrapper only proxies forward(), not custom
    # attributes, so `latent_encoder.num_components` would AttributeError once
    # prepared. Reading these from the raw module is also conceptually right:
    # they're per-run constants, not per-step state.
    num_components = latent_encoder.num_components
    num_registers = getattr(latent_encoder, "num_registers", 0)

    # Prepare everything with our `accelerator`.
    latent_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        latent_encoder, optimizer, train_dataloader, lr_scheduler
    )

    if train_unet:
        unet = accelerator.prepare(unet)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    accumulate_steps = 0  # necessary for args.gradient_accumulation_steps > 1

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(
                args.resume_from_checkpoint.rstrip("/")
            )  # only the checkpoint folder name is needed, not the full path
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            accumulate_steps = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        position=0,
        leave=True,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        if train_unet:
            unet.train()
        latent_encoder.train()
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the model input
            if args.offset_noise:
                noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                    model_input.shape[0],
                    model_input.shape[1],
                    1,
                    1,
                    device=model_input.device,
                )
            else:
                noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=model_input.device,
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            slot_tokens = latent_encoder(pixel_values)  # [B, K+R, D]

            if not train_unet:
                slot_tokens = slot_tokens.to(dtype=weight_dtype)

            model_pred = compose_eps(
                unet,
                noisy_model_input,
                timesteps,
                slot_tokens,
                num_components,
                num_registers,
            )

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Compute instance loss
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                base_weight = (
                    torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    / snr
                )

                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective needs to be floored to an SNR weight of one.
                    mse_loss_weights = base_weight + 1
                else:
                    # Epsilon and sample both use the same loss weights.
                    mse_loss_weights = base_weight
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            loss = loss / args.gradient_accumulation_steps

            # if args.with_prior_preservation:
            #     # Add the prior loss to the instance loss.
            #     loss = loss + args.prior_loss_weight * prior_loss

            accelerator.backward(loss)
            accumulate_steps += 1
            # if accelerator.sync_gradients:
            if (accumulate_steps + 1) % args.gradient_accumulation_steps == 0:
                params_to_clip = params_to_optimize
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if (accumulate_steps + 1) % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    images = []

                    if global_step % args.validation_steps == 0:
                        images = log_validation(
                            val_dataset=val_dataset,
                            latent_encoder=latent_encoder,
                            unet=unet,
                            vae=vae,
                            scheduler=noise_scheduler,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            global_step=global_step,
                            movi_eval_dataset=movi_eval_dataset,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-last")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
