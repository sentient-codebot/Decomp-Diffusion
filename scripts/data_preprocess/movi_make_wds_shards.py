"""Pack loose MOVi dumps into inode-efficient tar shards.

Input layout is the one written by ``movi_kubric_dump_with_labels.py``:

    movi-e-{split}-with-label/
      images/<video>/<frame>_image.png
      labels/<video>/<frame>_segment.png
      labels/<video>/<frame>_instances.json

Output layout:

    <output_root>/
      manifest.json
      train/
        manifest.json
        samples.jsonl
        movi-e-train-000000.tar
        ...

The tar members follow a WebDataset-style shared key convention:
``movi-e/train/<video>/<frame>.image.png``,
``.segment.png``, ``.instances.json``, and ``.meta.json``.
``samples.jsonl`` stores byte offsets so training can seek directly into the
tar without scanning tar metadata for every random sample.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tqdm.auto import tqdm


@dataclass(frozen=True)
class MoviSample:
    dataset: str
    split: str
    video: str
    frame: str
    image_path: Path
    segment_path: Path
    instances_path: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.split}/{self.video}/{self.frame}"

    @property
    def source_image_rel(self) -> str:
        return f"{self.video}/{self.frame}_image.png"


MEMBER_SUFFIXES = {
    "image": ".image.png",
    "segment": ".segment.png",
    "instances": ".instances.json",
    "meta": ".meta.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert loose MOVi PNG/JSON files into tar shards."
    )
    parser.add_argument(
        "--source_root",
        required=True,
        help="Loose MOVi root containing movi-e-{split}-with-label directories.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Destination root for shard directories and manifests.",
    )
    parser.add_argument("--dataset_name", default="movi-e")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=2048,
        help="Number of frames per tar shard.",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional cap for smoke-testing the converter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output_root after a successful temp build.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Count samples and check paths without writing shards.",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Do not remove the temporary output directory on failure.",
    )
    return parser.parse_args()


def iter_samples(source_root: Path, dataset: str, split: str):
    split_root = source_root / f"{dataset}-{split}-with-label"
    images_root = split_root / "images"
    labels_root = split_root / "labels"
    if not images_root.exists():
        raise FileNotFoundError(f"missing images root: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"missing labels root: {labels_root}")

    for image_path in sorted(images_root.glob("**/*_image.png")):
        rel = image_path.relative_to(images_root)
        frame = rel.name.removesuffix("_image.png")
        video = rel.parent.as_posix()
        segment_path = labels_root / rel.with_name(f"{frame}_segment.png")
        instances_path = labels_root / rel.with_name(f"{frame}_instances.json")
        missing = [
            str(path) for path in [segment_path, instances_path] if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"missing labels for {image_path}: {', '.join(missing)}"
            )
        yield MoviSample(
            dataset=dataset,
            split=split,
            video=video,
            frame=frame,
            image_path=image_path,
            segment_path=segment_path,
            instances_path=instances_path,
        )


def add_file(tar: tarfile.TarFile, path: Path, arcname: str):
    info = tar.gettarinfo(path, arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with open(path, "rb") as f:
        tar.addfile(info, f)


def add_bytes(tar: tarfile.TarFile, data: bytes, arcname: str):
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def write_shard(shard_path: Path, samples: list[MoviSample]):
    with tarfile.open(shard_path, "w") as tar:
        for sample in samples:
            members = {
                "image": f"{sample.key}.image.png",
                "segment": f"{sample.key}.segment.png",
                "instances": f"{sample.key}.instances.json",
                "meta": f"{sample.key}.meta.json",
            }
            add_file(tar, sample.image_path, members["image"])
            add_file(tar, sample.segment_path, members["segment"])
            add_file(tar, sample.instances_path, members["instances"])
            meta = {
                "dataset": sample.dataset,
                "split": sample.split,
                "video": sample.video,
                "frame": sample.frame,
                "source_image_rel": sample.source_image_rel,
            }
            add_bytes(
                tar,
                json.dumps(meta, sort_keys=True).encode("utf-8"),
                members["meta"],
            )


def index_shard(shard_path: Path) -> dict[str, dict[str, dict[str, int | str]]]:
    grouped: dict[str, dict[str, dict[str, int | str]]] = {}
    with tarfile.open(shard_path, "r:") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            for field, suffix in MEMBER_SUFFIXES.items():
                if member.name.endswith(suffix):
                    key = member.name[: -len(suffix)]
                    grouped.setdefault(key, {})[field] = {
                        "path": member.name,
                        "offset": member.offset_data,
                        "size": member.size,
                    }
                    break
    return grouped


def write_split(
    source_root: Path,
    split_root: Path,
    dataset: str,
    split: str,
    samples_per_shard: int,
    max_samples: int | None,
) -> dict[str, int | str]:
    split_root.mkdir(parents=True, exist_ok=True)
    index_path = split_root / "samples.jsonl"
    samples = list(iter_samples(source_root, dataset, split))
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(f"no samples found for {dataset} {split}")

    shard_count = 0
    with open(index_path, "w") as index_file:
        for start in tqdm(
            range(0, len(samples), samples_per_shard),
            desc=f"sharding {split}",
            unit="shard",
        ):
            shard_samples = samples[start : start + samples_per_shard]
            shard_name = f"{dataset}-{split}-{shard_count:06}.tar"
            shard_path = split_root / shard_name
            write_shard(shard_path, shard_samples)
            member_index = index_shard(shard_path)
            for sample in shard_samples:
                entry = {
                    "key": sample.key,
                    "dataset": sample.dataset,
                    "split": sample.split,
                    "video": sample.video,
                    "frame": sample.frame,
                    "source_image_rel": sample.source_image_rel,
                    "shard": shard_name,
                }
                entry.update(member_index[sample.key])
                index_file.write(json.dumps(entry, sort_keys=True) + "\n")
            shard_count += 1

    manifest = {
        "format": "movi-wds-v1",
        "dataset": dataset,
        "split": split,
        "samples": len(samples),
        "shards": shard_count,
        "samples_per_shard": samples_per_shard,
        "index": index_path.name,
        "source_root": os.fspath(source_root),
    }
    with open(split_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def count_split(source_root: Path, dataset: str, split: str, max_samples: int | None):
    count = 0
    for count, _sample in enumerate(iter_samples(source_root, dataset, split), start=1):
        if max_samples is not None and count >= max_samples:
            break
    return count


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if args.samples_per_shard <= 0:
        raise SystemExit("--samples_per_shard must be positive")

    if args.dry_run:
        for split in args.splits:
            count = count_split(
                source_root, args.dataset_name, split, args.max_samples_per_split
            )
            shards = (count + args.samples_per_shard - 1) // args.samples_per_shard
            print(f"{split}: {count} samples -> {shards} shards")
        return

    tmp_root = output_root.with_name(output_root.name + ".tmp")
    if tmp_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"temporary output exists: {tmp_root}; remove it or pass --overwrite"
            )
        shutil.rmtree(tmp_root)
    if output_root.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output_root}; pass --overwrite to replace")

    root_manifest = {
        "format": "movi-wds-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset_name,
        "source_root": os.fspath(source_root),
        "splits": {},
    }
    try:
        for split in args.splits:
            manifest = write_split(
                source_root=source_root,
                split_root=tmp_root / split,
                dataset=args.dataset_name,
                split=split,
                samples_per_shard=args.samples_per_shard,
                max_samples=args.max_samples_per_split,
            )
            root_manifest["splits"][split] = manifest
        with open(tmp_root / "manifest.json", "w") as f:
            json.dump(root_manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        if output_root.exists():
            shutil.rmtree(output_root)
        tmp_root.rename(output_root)
    except Exception:
        if not args.keep_tmp and tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise

    print(f"wrote {output_root}")
    for split, manifest in root_manifest["splits"].items():
        print(f"{split}: {manifest['samples']} samples in {manifest['shards']} shards")


if __name__ == "__main__":
    main()
