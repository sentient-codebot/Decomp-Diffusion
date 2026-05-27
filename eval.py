import argparse
import importlib
import os

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)

# from diffusers.training_utils import compute_snr # diffusers is still working on this, uncomment in future versions
from diffusers.utils import is_wandb_available
from PIL import Image
from torchvision.utils import make_grid

from src.data.dataset import GlobDataset
from src.models.encoder import load_latent_encoder
from src.pipeline.composable_stable_diffusion_pipeline import (
    ComposableStableDiffusionPipeline,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--pretrained_model_name",
    type=str,
    # Community mirror of stable-diffusion-2-1: stabilityai deprecated the
    # original repo, but the mirror carries the identical VAE weights.
    default="sd2-community/stable-diffusion-2-1",
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)

parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="Path to a checkpoint folder for the model.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=666,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
)

parser.add_argument(
    "--num_validation_images",
    type=int,
    default=32,
)

parser.add_argument(
    "--dataset_root",
    type=str,
    default=None,
)

parser.add_argument(
    "--dataset_glob",
    type=str,
    default=" **/*.jpg",
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
)

parser.add_argument(
    "--mixed_precision",
    type=str,
    default=None,
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)

parser.add_argument(
    "--validation_scheduler",
    type=str,
    default="DPMSolverMultistepScheduler",
    choices=["DPMSolverMultistepScheduler", "DDPMScheduler"],
    help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
)

parser.add_argument(
    "--scheduler_config",
    type=str,
    default=None,
    help="Path to a config file for the scheduler.",
    required=True,
)

parser.add_argument(
    "--resolution",
    type=int,
    default=256,
    help=(
        "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        " resolution"
    ),
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./logs/images",
    help="Path to a output folder for logging images.",
)

parser.add_argument(
    "--enable_xformers_memory_efficient_attention",
    action="store_true",
    help=(
        "Whether to use the memory efficient attention implementation of xFormers. This is an experimental feature"
        " and is only available for PyTorch >= 1.10.0 and xFormers >= 0.0.17."
    ),
)

args = parser.parse_args()


if is_wandb_available():
    pass

logger = get_logger(__name__)


@torch.no_grad()
def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # If passed along, set the training seed now.
    set_seed(args.seed)

    noise_scheduler_config = DDPMScheduler.load_config(args.scheduler_config)
    noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name, subfolder="vae")
    vae.to(accelerator.device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        args.ckpt_path, subfolder="unet2dconditionmodel".lower()
    )
    unet = unet.to(device=accelerator.device, dtype=weight_dtype)
    print("loaded a trained unet2dconditionmodel")
    # Detects the encoder type (LatentEncoder vs SlotAttentionEncoder) from the
    # checkpoint subfolder, so eval works for either training run.
    latent_encoder = load_latent_encoder(args.ckpt_path)
    latent_encoder = latent_encoder.to(device=accelerator.device, dtype=weight_dtype)
    print(f"loaded a trained {type(latent_encoder).__name__}")

    # prepare validation data
    val_dataset = GlobDataset(
        root=args.dataset_root,
        img_size=args.resolution,
        img_glob=args.dataset_glob,
        data_portion=(0.0, 1.0),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    print("***** Running testing *****")
    print(f"  Num examples = {len(val_dataset)}")
    print(f"  Num batches each epoch = {len(val_dataloader)}")

    unet = accelerator.unwrap_model(unet)
    latent_encoder = accelerator.unwrap_model(latent_encoder)

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in noise_scheduler.config:
        variance_type = noise_scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    # use a more efficient scheduler at test time
    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    scheduler = scheduler_class.from_config(noise_scheduler.config, **scheduler_args)

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

    image_log_dir = "./image_test_output/"
    os.makedirs(image_log_dir, exist_ok=True)

    images = []
    image_count = 0
    print("start testing the model on data")

    for batch_idx, batch in enumerate(val_dataloader):
        print(batch_idx)

        pixel_values = batch["pixel_values"].to(
            device=accelerator.device, dtype=weight_dtype
        )

        with torch.autocast("cuda"):
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
            # Per-slot decode: under the mean-of-eps training objective each
            # per-slot eps is already on a unit-noise scale, so guidance_scale=1
            # matches what the model saw during training.
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
                    guidance_scale=1.0,
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


if __name__ == "__main__":
    main(args)
