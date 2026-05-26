import glob
import json
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        self.transform = transforms.Compose(
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

        if vit_norm:
            self.transform_vit = transforms.Compose(
                [
                    transforms.Resize(
                        vit_input_resolution,
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.CenterCrop(vit_input_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

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
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
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


if __name__ == "__main__":
    dataset = GlobDataset(
        root="/research/projects/object_centric/shared_datasets/movi/movi-e/movi-e-train-with-label/images/",
        img_size=256,
        img_glob="**/*.png",
        data_portion=(0.0, 0.9),
    )
    pass
