# MOVi-E encoder configs -- side by side

Reference table comparing the three encoder front ends configured for MOVi-E.
All three project to `num_components=11` slots of `latent_dim=64` for the UNet
cross-attention; they differ only in what produces the feature map fed into
the slot read-out.

| Encoder | Config | Input image | Feature map (H x W) | Feature dim |
|---|---|---|---|---|
| Plain CNN (`LatentEncoder` / `SlotAttentionEncoder`) | `configs/movi-e/{latent,slot}_encoder/config.json` | 128 x 128 | 16 x 16 (3 stride-2 convs from 128) | 1024 (`enc_channels=128` x 2^3) |
| DINO v1 (`facebook/dino-vits8`) | `configs/movi-e/dino_slot_encoder/config.json` | 128 x 128 | 16 x 16 (patch=8 -> 128/8) | 384 (ViT-S hidden) |
| DINO v3 (`facebook/dinov3-vits16-pretrain-lvd1689m`) | `configs/movi-e/dinov3_slot_encoder/config.json` | 256 x 256 | 16 x 16 (patch=16 -> 256/16) | 384 (ViT-S hidden) |

Notes:
- Both DINO configs were sized to land on the same 16 x 16 token grid as the
  CNN baseline -- DINOv3 bumps the image to 256 so that patch=16 still yields
  16 x 16.
- For `LatentEncoder`, the 16 x 16 x 1024 map exists internally but is
  immediately flattened + linearly read out into the 11 slots (no slot
  attention); for the other two it's positional-embedded and fed to Slot
  Attention (`src/models/encoder.py:166`, `src/models/encoder.py:238`).
- `configs/movi-e/train_config.yaml` sets `resolution: 128`, so the DINOv3
  run additionally needs `--resolution 256` on the launch line to match its
  `image_size=256`.
