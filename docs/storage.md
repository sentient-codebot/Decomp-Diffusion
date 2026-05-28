# Storage and inode policy

Persistent project storage lives under `~/prjs0993`. Keep high-file-count raw
layouts off this filesystem unless they are temporary and actively being
converted.

## MOVi-E

The canonical MOVi-E layout is now WebDataset-style tar shards under:

```bash
data/movi-e-wds/{train,validation,test}/
```

Each split has:

- `*.tar` shards containing image, segment, instance JSON, and metadata members.
- `samples.jsonl` with one line per frame and byte offsets into the shard.
- `manifest.json` with split-level counts.

To convert an existing loose dump:

```bash
sbatch jobs/movi_e_shard_wds.sh
```

The sharding job defaults to `SOURCE_ROOT=data/movi-e` and
`OUTPUT_ROOT=data/movi-e-wds`. If the loose dump is on scratch, override the
source at submission time:

```bash
SOURCE_ROOT=/scratch-shared/nlin/movi-e-loose-<job>/movi-e sbatch jobs/movi_e_shard_wds.sh
```

Future TFDS preprocessing should use `jobs/movi_e_preprocess.sh`, which writes
the loose PNG/JSON expansion under `/scratch-shared/nlin/...` and publishes only
`data/movi-e-wds` to `prjs0993`.

After shard-based training and MOVi metrics pass, remove the persistent loose
`~/prjs0993/datasets/movi-e/` tree. Keep scratch copies only for manual
inspection, assuming they may be purged.

## Other datasets

COCO and CelebA-HQ remain loose files for now. COCO should be sharded if repeated
natural-image experiments become common; CelebA-HQ is small enough to defer.

## Logs and caches

Keep reports, final checkpoints, shard manifests, and synced W&B run URLs.
Remove redundant local W&B run directories after sync unless a run is needed for
offline debugging. Keep `~/prjs0993/tmp/torchinductor`; it protects warm
`torch.compile` startup time.

Run the read-only inode report before large launches:

```bash
bash scripts/storage/inode_report.sh
```
