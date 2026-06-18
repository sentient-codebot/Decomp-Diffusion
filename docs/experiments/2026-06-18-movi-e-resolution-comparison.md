# MOVi-E resolution comparison

This note tracks the resolution and dimensionality choices that matter for
comparing the current MOVi-E pipeline against recent object-centric diffusion
work. The main comparison target is CODA 2026, but the table also includes the
closest older diffusion and feature-reconstruction references.

| Paper / run | Source image size | Model input size | Backbone | Patch size | Slot-attn feature grid | Feature dim | K | R | Slot dim | Diffusion base | VAE / downsample | Diffusion latent grid / channels | Notes / source |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| CODA 2026 | MOVi source, resized | 512 x 512 | DINOv2 ViT-B/14 | 14 | 32 x 32 | 768 | 24 on MOVi-E | 77 frozen CLIP padding tokens | 768 | SD1.5 | SD VAE, 8x | 64 x 64 x 4 | Paper says CODA extracts slots from DINOv2 features and decodes with pretrained SD1.5. Repo config uses `dino_sample_size=32`, `dino_out_channels=768`, `slot_size=768`, SD1.5, and K/V/O cross-attn finetuning. Sources: [paper](https://arxiv.org/html/2601.01224v2), [repo](https://github.com/sony/coda), [encoder config](https://github.com/sony/coda/blob/main/experiment/setup/pipeline/encoder/lsd_register.yaml), [pipeline config](https://github.com/sony/coda/blob/main/experiment/setup/pipeline/pipeline.yaml). |
| SlotDiffusion on MOVi-E | MOVi frames | 128 x 128 | SAVi/CNN encoder | n/a | model-specific | 192 slot size | 15 | 0 | 192 | LDM trained from scratch over VQ-VAE latents | VQ-VAE, 4x | 32 x 32 x 3 | The MOVi-E config is named `res128`, sets `resolution=(128, 128)`, `slot_size=192`, `num_slots=15`, `latent_ch=3`, and decoder resolution `res // 4`. Sources: [docs](https://raw.githubusercontent.com/Wuziyi616/SlotDiffusion/main/docs/video_based.md), [config](https://raw.githubusercontent.com/Wuziyi616/SlotDiffusion/main/slotdiffusion/video_based/configs/savi_ldm/savi_ldm_movie_params-res128.py). |
| Stable-LSD / LSD | not fully sourceable here | not fully sourceable here | Slot encoder + Stable Diffusion decoder | not fully sourceable here | not fully sourceable here | not fully sourceable here | not fully sourceable here | not fully sourceable here | not fully sourceable here | Stable Diffusion | SD VAE, 8x when using SD | image / 8 x image / 8 x 4 | CODA cites Stable-LSD as a diffusion-OCL baseline using pretrained diffusion. Exact MOVi-E resolution/config should be filled from the original implementation or paper before making numeric claims. Source for context: [CODA related work](https://arxiv.org/html/2601.01224v2). |
| DINOSAUR on MOVi-E | MOVi frames | 224 x 224 DINO path; masks evaluated at image resolution | DINO ViT-S/8 or ViT-B/8 on synthetic datasets | 8 | 28 x 28 | 384 for ViT-S, 768 for ViT-B | paper/config dependent | 0 | paper/config dependent | none | none | none | DINOSAUR reconstructs DINO features instead of image pixels or diffusion latents. The paper states synthetic experiments use ViT patch size 8; the public MOVi-E config uses `vit_small_patch8_224_dino` with `num_patches=784`, i.e. 28 x 28. Sources: [paper](https://arxiv.org/abs/2209.14860), [OCLF config](https://amazon-science.github.io/object-centric-learning-framework/configs/experiment/projects/bridging/dinosaur/movi_e_feat_rec/). |
| This repo: current MOVi-E DINOv3 K/V-only | MOVi-E TFDS 256 x 256 | 256 x 256 | DINOv3 ViT-S/16 | 16 | 16 x 16 | 384 | 24 | 4 learned | 1024 | SD2.1 frozen except cross-attn K/V | SD2.1 VAE, 8x | 32 x 32 x 4 | Current controlled CoDA-style run: `jobs/movi_e_coda_kv_only_train_eval.sh` with `configs/movi-e/dinov3_slot_encoder_d1024/config.json`. |
| This repo: planned MOVi-E DINOv3 512 K/V-only | MOVi-E TFDS 256 x 256 | 512 x 512 upsampled | DINOv3 ViT-S/16 | 16 | 32 x 32 | 384 | 24 | 4 learned | 1024 | SD2.1 frozen except cross-attn K/V | SD2.1 VAE, 8x | 64 x 64 x 4 | New controlled comparison run: `jobs/movi_e_coda_kv_only_512_train_eval.sh` with `configs/movi-e/dinov3_slot_encoder_d1024_512/config.json`. Closes the image/grid gap to CODA, but not the DINOv2, SD1.5, 768-slot, CLIP-register, or contrastive-loss gaps. |

## Takeaways

- The current 256 DINOv3 run and the planned 512 DINOv3 run both use MOVi-E
  frames stored at 256 x 256. The planned run upsamples frames to 512 x 512
  before DINOv3 and VAE encoding.
- The planned run changes two spatial grids at once: DINOv3 slot-attention
  input goes from 16 x 16 to 32 x 32, and SD2.1 diffusion latents go from
  32 x 32 x 4 to 64 x 64 x 4.
- The planned run is a controlled extension of our current recipe. It should be
  described as closing CODA's resolution/grid gap, not as reproducing CODA.
- LDM-style diffusion latents use a spatial downsample factor `f=H/h=W/w`; with
  SD-style 8x VAE compression, 256 input gives 32 x 32 latents and 512 input
  gives 64 x 64 latents. Source: [LDM paper](https://arxiv.org/abs/2112.10752).
