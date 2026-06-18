# MOVi-E encoder configs -- side by side

Reference table comparing the encoder front ends configured for MOVi-E. The
important distinction is between the raw/model input resolution, the
slot-attention feature grid, and the SD-VAE diffusion latent grid.

| Encoder / run family | Config | Model input | Feature map (H x W) | Feature dim | Object slots K | Registers R | Slot dim | Diffusion latent grid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Plain CNN (`LatentEncoder` / `SlotAttentionEncoder`) | `configs/movi-e/{latent,slot}_encoder/config.json` | 128 x 128 | 16 x 16 (3 stride-2 convs from 128) | 1024 (`enc_channels=128` x 2^3) | 11 | 0 | 64 | 16 x 16 x 4 |
| DINO v1 (`facebook/dino-vits8`) | `configs/movi-e/dino_slot_encoder/config.json` | 128 x 128 | 16 x 16 (patch=8 -> 128/8) | 384 (ViT-S hidden) | 11 | 0 | 64 | 16 x 16 x 4 |
| DINOv3 controlled / register run | `configs/movi-e/dinov3_slot_encoder/config.json` | 256 x 256 | 16 x 16 (patch=16 -> 256/16) | 384 (ViT-S hidden) | 24 | 4 | 64 | 32 x 32 x 4 |
| DINOv3 CoDA-style K/V-only run | `configs/movi-e/dinov3_slot_encoder_d1024/config.json` | 256 x 256 | 16 x 16 (patch=16 -> 256/16) | 384 (ViT-S hidden) | 24 | 4 | 1024 | 32 x 32 x 4 |
| DINOv3 CoDA-style K/V-only 512 run | `configs/movi-e/dinov3_slot_encoder_d1024_512/config.json` | 512 x 512 | 32 x 32 (patch=16 -> 512/16) | 384 (ViT-S hidden) | 24 | 4 | 1024 | 64 x 64 x 4 |

Notes:
- MOVi-E is preprocessed from the TFDS `movi_e/256x256:1.0.0` release. Any
  512 input run upsamples the stored 256 x 256 frames through the dataset
  transform before the DINO backbone and SD VAE see them.
- `latent_dim` is the slot-token dimension used as UNet cross-attention
  conditioning. It is not the SD-VAE latent channel count. The SD2.1 VAE latent
  is always 4 channels with an 8x spatial downsample.
- For `LatentEncoder`, the 16 x 16 x 1024 map exists internally but is
  immediately flattened and linearly read out into slots. The slot-attention
  encoders instead keep the spatial feature map, positional-embed it, and feed
  it to Slot Attention.
- `configs/movi-e/train_config.yaml` still defaults to `resolution: 128`.
  DINOv3 launch scripts must override this with `--resolution 256` or
  `--resolution 512` to match their encoder `image_size`.
