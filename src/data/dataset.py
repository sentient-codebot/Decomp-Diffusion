import glob
import io
import json
import os
import random
import tarfile
from collections import OrderedDict
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MOVI_WDS_FIELDS = {
    "image": ".image.png",
    "segment": ".segment.png",
    "instances": ".instances.json",
    "meta": ".meta.json",
}


def _as_list(value):
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def _expand_data_portions(data_portion, n):
    if (
        not all(hasattr(sublist, "__iter__") for sublist in data_portion)
        or data_portion == ()
    ):
        return [data_portion] * n
    return list(data_portion)


def _apply_data_portion(samples, data_portion, random_data_on_portion):
    assert len(data_portion) == 0 or len(data_portion) == 2
    if len(data_portion) == 2:
        assert max(data_portion) <= 1.0 and min(data_portion) >= 0.0

    if data_portion and data_portion != (0.0, 1.0):
        samples = list(samples)
        if random_data_on_portion:
            random.Random(42).shuffle(samples)
        samples = samples[
            int(len(samples) * data_portion[0]) : int(len(samples) * data_portion[1])
        ]
    return samples


def _image_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize(
                img_size,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def _vit_transform(vit_input_resolution):
    return transforms.Compose(
        [
            transforms.Resize(
                vit_input_resolution,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(vit_input_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class _ShardFileCache:
    """Small LRU of open shard files, local to each dataloader worker."""

    def __init__(self, max_open=16):
        self.max_open = max_open
        self._files = OrderedDict()

    def read(self, shard_path, member):
        shard_path = os.fspath(shard_path)
        if isinstance(member, dict) and "offset" in member and "size" in member:
            f = self._file(shard_path)
            f.seek(int(member["offset"]))
            return f.read(int(member["size"]))

        # Compatibility fallback for hand-built shards without byte offsets.
        with tarfile.open(shard_path, "r:") as tar:
            extracted = tar.extractfile(
                member["path"] if isinstance(member, dict) else member
            )
            if extracted is None:
                raise KeyError(f"missing tar member {member} in {shard_path}")
            return extracted.read()

    def _file(self, shard_path):
        f = self._files.pop(shard_path, None)
        if f is None:
            f = open(shard_path, "rb")
        self._files[shard_path] = f
        while len(self._files) > self.max_open:
            _, old = self._files.popitem(last=False)
            old.close()
        return f

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()


def _field_path(member):
    if isinstance(member, dict):
        return member.get("path", "")
    return member


def _sample_matches(sample, pattern):
    if pattern in (None, "", "*.tar"):
        return True
    candidates = [
        _field_path(sample.get("image", "")),
        sample.get("source_image_rel", ""),
    ]
    # Compatibility with the loose-file MOVi glob convention.
    if sample.get("video") and sample.get("frame"):
        candidates.append(f"{sample['video']}/{sample['frame']}_image.png")
    return any(fnmatch(candidate, pattern) for candidate in candidates if candidate)


def _scan_shard_samples(shard_paths):
    samples = []
    for shard_path in shard_paths:
        grouped = {}
        with tarfile.open(shard_path, "r:") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                for field, suffix in MOVI_WDS_FIELDS.items():
                    if member.name.endswith(suffix):
                        key = member.name[: -len(suffix)]
                        grouped.setdefault(key, {})[field] = {
                            "path": member.name,
                            "offset": member.offset_data,
                            "size": member.size,
                        }
                        break
        for key, entry in sorted(grouped.items()):
            entry["key"] = key
            entry["shard"] = os.path.basename(shard_path)
            entry["_shard_path"] = os.fspath(shard_path)
            parts = key.split("/")
            if len(parts) >= 4:
                entry["dataset"] = parts[-4]
                entry["split"] = parts[-3]
                entry["video"] = parts[-2]
                entry["frame"] = parts[-1]
                entry["source_image_rel"] = f"{parts[-2]}/{parts[-1]}_image.png"
            samples.append(entry)
    return samples


def _load_wds_samples(root, img_glob="*.tar", split=None):
    root = Path(root)
    shard_paths = sorted(root.glob(img_glob))
    sample_filter = None
    if not shard_paths:
        # Treat non-tar globs as sample filters and read all split shards.
        sample_filter = img_glob
        shard_paths = sorted(root.glob("*.tar"))
    shard_paths = [p for p in shard_paths if p.is_file()]
    allowed_shards = {p.name for p in shard_paths}

    index_path = root / "samples.jsonl"
    if not index_path.exists():
        if not shard_paths:
            raise FileNotFoundError(f"No tar shards found under {root}")
        samples = _scan_shard_samples(shard_paths)
    else:
        samples = []
        with open(index_path) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if allowed_shards and sample["shard"] not in allowed_shards:
                    continue
                sample["_shard_path"] = os.fspath(root / sample["shard"])
                samples.append(sample)

    if split is not None:
        samples = [sample for sample in samples if sample.get("split") == split]
    if sample_filter is not None:
        samples = [
            sample for sample in samples if _sample_matches(sample, sample_filter)
        ]
    if not samples:
        raise FileNotFoundError(f"No shard samples found under {root}")
    return samples


def _load_image_from_bytes(data):
    image = Image.open(io.BytesIO(data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    else:
        image.load()
    return image


class GlobDataset(Dataset):
    def __init__(
        self,
        root,
        img_size,
        img_glob="*.png",
        data_portion=(),
        random_data_on_portion=True,
        vit_norm=False,
        random_flip=False,
        vit_input_resolution=448,
    ):
        super().__init__()
        if isinstance(root, str) or not hasattr(root, "__iter__"):
            root = [root]
            img_glob = [img_glob]
        if (
            not all(hasattr(sublist, "__iter__") for sublist in data_portion)
            or data_portion == ()
        ):  # if not iterable or empty
            data_portion = [data_portion]
        self.root = root
        self.img_size = img_size
        self.episodes = []
        self.vit_norm = vit_norm
        self.random_flip = random_flip

        for n, (r, g) in enumerate(zip(root, img_glob, strict=False)):
            episodes = glob.glob(os.path.join(r, g), recursive=True)

            episodes = sorted(episodes)

            data_p = data_portion[n]

            assert len(data_p) == 0 or len(data_p) == 2
            if len(data_p) == 2:
                assert max(data_p) <= 1.0 and min(data_p) >= 0.0

            if data_p and data_p != (0.0, 1.0):
                if random_data_on_portion:
                    random.Random(42).shuffle(episodes)  # fix results
                episodes = episodes[
                    int(len(episodes) * data_p[0]) : int(len(episodes) * data_p[1])
                ]

            self.episodes += episodes

        # resize the shortest side to img_size and center crop
        self.transform = _image_transform(img_size)

        if vit_norm:
            self.transform_vit = _vit_transform(vit_input_resolution)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.episodes[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.random_flip:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        pixel_values = self.transform(image)
        example["pixel_values"] = pixel_values
        if self.vit_norm:
            image_vit = self.transform_vit(image)
            example["pixel_values_vit"] = image_vit
        return example


class WdsImageDataset(Dataset):
    """Reads WebDataset-style MOVi tar shards as a map-style image dataset."""

    def __init__(
        self,
        root,
        img_size,
        img_glob="*.tar",
        data_portion=(),
        random_data_on_portion=True,
        vit_norm=False,
        random_flip=False,
        vit_input_resolution=448,
    ):
        super().__init__()
        roots = _as_list(root)
        globs = _as_list(img_glob)
        if len(globs) == 1 and len(roots) > 1:
            globs *= len(roots)
        data_portions = _expand_data_portions(data_portion, len(roots))

        self.samples = []
        for r, g, data_p in zip(roots, globs, data_portions, strict=False):
            samples = _load_wds_samples(r, g)
            self.samples += _apply_data_portion(samples, data_p, random_data_on_portion)

        self.img_size = img_size
        self.vit_norm = vit_norm
        self.random_flip = random_flip
        self.transform = _image_transform(img_size)
        if vit_norm:
            self.transform_vit = _vit_transform(vit_input_resolution)
        self._file_cache = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_cache"] = None
        return state

    def __len__(self):
        return len(self.samples)

    def _read_member(self, sample, field):
        if self._file_cache is None:
            self._file_cache = _ShardFileCache()
        return self._file_cache.read(sample["_shard_path"], sample[field])

    def __getitem__(self, i):
        example = {}
        image = _load_image_from_bytes(self._read_member(self.samples[i], "image"))
        if self.random_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        example["pixel_values"] = self.transform(image)
        if self.vit_norm:
            example["pixel_values_vit"] = self.transform_vit(image)
        return example

    def __del__(self):
        if getattr(self, "_file_cache", None) is not None:
            self._file_cache.close()


class MoviPairDataset(Dataset):
    """Reads MOVi-E (image, segment[, properties]) tuples from the dumped layout.

    The dump (`scripts/data_preprocess/movi_kubric_dump_with_labels.py`) writes
    one `<frame>_image.png` + `<frame>_segment.png` (+ `<frame>_instances.json`)
    per frame under `<root>/movi-e-<split>-with-label/{images,labels}/<vid>/`.

    When `load_properties=True`, also returns per-frame instance attributes
    (`category`, `image_positions`, `bboxes_3d`) padded to `max_instances` so
    the loader can collate variable-length per-frame counts. A `valid` mask
    indicates which rows are real vs. padding.
    """

    def __init__(
        self,
        root,
        split,
        img_size,
        max_images=None,
        load_properties=False,
        max_instances=24,
    ):
        self.img_size = img_size
        images_root = os.path.join(root, f"movi-e-{split}-with-label", "images")
        labels_root = os.path.join(root, f"movi-e-{split}-with-label", "labels")
        image_paths = sorted(
            glob.glob(os.path.join(images_root, "**", "*_image.png"), recursive=True)
        )
        if not image_paths:
            raise FileNotFoundError(f"No frames found under {images_root}")
        if max_images is not None:
            image_paths = image_paths[:max_images]
        self.image_paths = image_paths
        self.labels_root = labels_root
        self.images_root = images_root
        self.load_properties = load_properties
        self.max_instances = max_instances
        # Image transform: same normalisation as training (GlobDataset).
        self.image_transform = _image_transform(img_size)
        # Segment transform: nearest-neighbour so integer instance ids survive.
        self.segment_transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(img_size),
            ]
        )

    def _label_path(self, image_path, suffix):
        # images/<vid>/<frame>_image.png  ->  labels/<vid>/<frame>_<suffix>
        rel = os.path.relpath(image_path, self.images_root)
        rel = rel.replace("_image.png", suffix)
        return os.path.join(self.labels_root, rel)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert("RGB")
        seg = Image.open(self._label_path(self.image_paths[i], "_segment.png"))
        pixel_values = self.image_transform(img)
        seg_resized = self.segment_transform(seg)
        segment = torch.from_numpy(np.array(seg_resized, dtype=np.int64))
        sample = {"pixel_values": pixel_values, "segment": segment}
        if self.load_properties:
            sample.update(self._load_properties(self.image_paths[i]))
        return sample

    def _load_properties(self, image_path):
        with open(self._label_path(image_path, "_instances.json")) as f:
            inst = json.load(f)
        n_max = self.max_instances
        category = np.zeros((n_max,), dtype=np.int64)
        image_positions = np.zeros((n_max, 2), dtype=np.float32)
        bboxes_3d = np.zeros((n_max, 8, 3), dtype=np.float32)
        valid = np.zeros((n_max,), dtype=bool)

        # MOVi instance ids in the segment PNG are 1-indexed; index 0 is bg.
        cat = np.asarray(inst.get("category", []), dtype=np.int64)
        pos = np.asarray(inst.get("image_positions", []), dtype=np.float32)
        bbox = np.asarray(inst.get("bboxes_3d", []), dtype=np.float32)
        n = min(len(cat), n_max)
        if n > 0:
            category[:n] = cat[:n]
            valid[:n] = True
            if pos.size and pos.shape[0] >= n:
                image_positions[:n] = pos[:n]
            if bbox.size and bbox.shape[0] >= n:
                bboxes_3d[:n] = bbox[:n]
        return {
            "category": torch.from_numpy(category),
            "image_positions": torch.from_numpy(image_positions),
            "bboxes_3d": torch.from_numpy(bboxes_3d),
            "valid": torch.from_numpy(valid),
        }


class MoviWdsPairDataset(Dataset):
    """Reads MOVi-E image/segment/property tuples from tar shards."""

    def __init__(
        self,
        root,
        split,
        img_size,
        max_images=None,
        load_properties=False,
        max_instances=24,
    ):
        root = Path(root)
        split_root = root / split
        if not split_root.exists():
            split_root = root
        samples = _load_wds_samples(split_root, "*.tar", split=split)
        if max_images is not None:
            samples = samples[:max_images]

        self.samples = samples
        self.img_size = img_size
        self.load_properties = load_properties
        self.max_instances = max_instances
        self.image_transform = _image_transform(img_size)
        self.segment_transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(img_size),
            ]
        )
        self._file_cache = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_cache"] = None
        return state

    def __len__(self):
        return len(self.samples)

    def _read_member(self, sample, field):
        if self._file_cache is None:
            self._file_cache = _ShardFileCache()
        return self._file_cache.read(sample["_shard_path"], sample[field])

    def __getitem__(self, i):
        sample_info = self.samples[i]
        img = _load_image_from_bytes(self._read_member(sample_info, "image"))
        seg = Image.open(io.BytesIO(self._read_member(sample_info, "segment")))
        pixel_values = self.image_transform(img)
        seg_resized = self.segment_transform(seg)
        segment = torch.from_numpy(np.array(seg_resized, dtype=np.int64))
        sample = {"pixel_values": pixel_values, "segment": segment}
        if self.load_properties:
            inst = json.loads(self._read_member(sample_info, "instances").decode())
            sample.update(self._properties_to_tensors(inst))
        return sample

    def _properties_to_tensors(self, inst):
        n_max = self.max_instances
        category = np.zeros((n_max,), dtype=np.int64)
        image_positions = np.zeros((n_max, 2), dtype=np.float32)
        bboxes_3d = np.zeros((n_max, 8, 3), dtype=np.float32)
        valid = np.zeros((n_max,), dtype=bool)

        cat = np.asarray(inst.get("category", []), dtype=np.int64)
        pos = np.asarray(inst.get("image_positions", []), dtype=np.float32)
        bbox = np.asarray(inst.get("bboxes_3d", []), dtype=np.float32)
        n = min(len(cat), n_max)
        if n > 0:
            category[:n] = cat[:n]
            valid[:n] = True
            if pos.size and pos.shape[0] >= n:
                image_positions[:n] = pos[:n]
            if bbox.size and bbox.shape[0] >= n:
                bboxes_3d[:n] = bbox[:n]
        return {
            "category": torch.from_numpy(category),
            "image_positions": torch.from_numpy(image_positions),
            "bboxes_3d": torch.from_numpy(bboxes_3d),
            "valid": torch.from_numpy(valid),
        }

    def __del__(self):
        if getattr(self, "_file_cache", None) is not None:
            self._file_cache.close()


def build_image_dataset(dataset_format, **kwargs):
    if dataset_format == "files":
        return GlobDataset(**kwargs)
    if dataset_format == "wds":
        return WdsImageDataset(**kwargs)
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def build_movi_pair_dataset(dataset_format, **kwargs):
    if dataset_format == "files":
        return MoviPairDataset(**kwargs)
    if dataset_format == "wds":
        return MoviWdsPairDataset(**kwargs)
    raise ValueError(f"Unsupported MOVi dataset format: {dataset_format}")


if __name__ == "__main__":
    dataset = GlobDataset(
        root="/research/projects/object_centric/shared_datasets/movi/movi-e/movi-e-train-with-label/images/",
        img_size=256,
        img_glob="**/*.png",
        data_portion=(0.0, 0.9),
    )
    pass
