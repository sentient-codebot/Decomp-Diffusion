import argparse
import os

import yaml


def parse_args(input_args=None):
    # `--train_config` is resolved first by a bare parser so its YAML contents
    # can be folded in as defaults before the full command line is parsed.
    # Sharing it via `parents=` keeps it visible in `--help`.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help=(
            "Path to a YAML file of training hyperparameters. Its values act as "
            "defaults; any flag passed on the command line overrides them."
        ),
    )

    parser = argparse.ArgumentParser(
        description="Simple example of a training script.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        # Community mirror of stable-diffusion-2-1: stabilityai deprecated the
        # original repo, but the mirror carries the identical VAE weights.
        default="sd2-community/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="log",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )  # for reference: for movi-e experiment, each pack of ckpt takes roughly 3.9GB. model weights are only < 1.4g though, just the optimizer state is huge 2.6g
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    # parser.add_argument(
    #     "--hub_model_id",
    #     type=str,
    #     default=None,
    #     help="The name of the repository to keep in sync with the local `output_dir`.",
    # ) # not supported yet
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
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
    )  # recommended

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )

    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="DPMSolverMultistepScheduler",
        choices=["DPMSolverMultistepScheduler", "DDPMScheduler"],
        help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )

    # ------------------------------------ Latent slot diffusion ------------------------------------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="latent_decomposed_diffusion",
        help="The name of the tracker project to use for logging.",
    )
    parser.add_argument(
        "--latent_encoder_config",
        type=str,
        default=None,
        help="Path to a config file for the slot attention.",
        required=True,
    )
    parser.add_argument(
        "--unet_config",
        type=str,
        default=None,
        help="Path to a config file for the unet or pretrain_sd.",
        required=True,
    )
    parser.add_argument(
        "--freeze_unet_except_kv",
        action="store_true",
        help=(
            "Only valid with --unet_config pretrain_sd. Keep the pretrained UNet "
            "frozen except for cross-attention to_k / to_v projections (CoDA-style)."
        ),
    )
    parser.add_argument(
        "--warm_start_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint folder to load model weights from before training. "
            "Optimizer, scheduler, and global step are not restored; if "
            "--resume_from_checkpoint resolves inside --output_dir, resume wins."
        ),
    )
    parser.add_argument(
        "--epsilon_composition",
        type=str,
        default="mean",
        choices=["mean", "slot_attn", "slot_attn_pool"],
        help=(
            "How to compose per-slot epsilon predictions. 'mean' uses uniform "
            "1/K weights; 'slot_attn' reuses detached encoder slot-attention "
            "masks interpolated to the epsilon resolution; 'slot_attn_pool' "
            "spatially pools those masks to one scalar weight per slot."
        ),
    )
    parser.add_argument(
        "--scheduler_config",
        type=str,
        default=None,
        help="Path to a config file for the scheduler.",
        required=True,
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to the dataset root.",
        required=True,
    )
    parser.add_argument(
        "--dataset_glob",
        type=str,
        default="**/*.png",
        help="Glob pattern for the dataset.",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="files",
        choices=["files", "wds"],
        help=(
            "Dataset storage format for --dataset_root. 'files' reads loose "
            "images; 'wds' reads MOVi WebDataset-style tar shards."
        ),
    )
    parser.add_argument(
        "--encoder_lr_scale",
        type=float,
        default=3.0,
        help="Scale the learning rate of the encoder by this factor. 1.0 means same learning rate as the ldm model.",
    )

    parser.add_argument(
        "--flip_images",
        action="store_true",
        help="Whether to flip the image horizontally in training.",
    )
    parser.add_argument(
        "--movi_eval_root",
        type=str,
        default=None,
        help=(
            "Optional path to a MOVi-E dump root (containing "
            "`movi-e-{split}-with-label/{images,labels}/`). When set and the "
            "encoder exposes slot attention masks, val/fg_ari, val/mbo, "
            "val/miou are logged each validation step. Off by default so "
            "non-MOVi-E runs are unaffected."
        ),
    )
    parser.add_argument(
        "--movi_eval_format",
        type=str,
        default="files",
        choices=["files", "wds"],
        help="Storage format for --movi_eval_root.",
    )
    parser.add_argument(
        "--movi_eval_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--movi_eval_max_images",
        type=int,
        default=256,
        help="Cap on frames used for in-training segmentation metrics.",
    )
    parser.add_argument(
        "--train_split_portion",
        type=float,
        default=0.9,
        help="Portion of the dataset to use for training.",
    )
    # Fold the YAML hyperparameter file in as argparse defaults so that any
    # value also given on the command line still takes precedence.
    cfg_args, _ = config_parser.parse_known_args(input_args)
    if cfg_args.train_config is not None:
        with open(cfg_args.train_config) as f:
            config = yaml.safe_load(f) or {}
        valid_keys = {action.dest for action in parser._actions}
        unknown = set(config) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown keys in {cfg_args.train_config}: {sorted(unknown)}"
            )
        parser.set_defaults(**config)

    args = parser.parse_args(input_args)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.output_dir = os.path.join(args.output_dir, args.tracker_project_name)

    return args


if __name__ == "__main__":
    # define and save scheduler config from here
    import os
    import sys

    if __name__ == "__main__":
        sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
    from diffusers.schedulers import DDPMScheduler

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
        beta_schedule="linear",
    )
    scheduler.save_config("./configs/movi-e/scheduler")
