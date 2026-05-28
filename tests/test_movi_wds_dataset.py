"""Smoke tests for MOVi WebDataset-style shard conversion and reading.

Run with: `uv run python -m unittest tests/test_movi_wds_dataset.py`
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import MoviWdsPairDataset, WdsImageDataset


class MoviWdsDatasetTest(unittest.TestCase):
    def _write_loose_sample(self, root, split, video, frame, value, segment_id):
        images_dir = root / f"movi-e-{split}-with-label" / "images" / video
        labels_dir = root / f"movi-e-{split}-with-label" / "labels" / video
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image = np.full((8, 8, 3), value, dtype=np.uint8)
        segment = np.full((8, 8), segment_id, dtype=np.uint8)
        Image.fromarray(image).save(images_dir / f"{frame}_image.png")
        Image.fromarray(segment).save(labels_dir / f"{frame}_segment.png")
        instances = {
            "category": [3],
            "image_positions": [[0.25, 0.75]],
            "bboxes_3d": [[[float(i + j * 3) for i in range(3)] for j in range(8)]],
        }
        with open(labels_dir / f"{frame}_instances.json", "w") as f:
            json.dump(instances, f)

    def test_convert_and_read_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "loose" / "movi-e"
            output = tmp_path / "movi-e-wds"
            self._write_loose_sample(source, "train", "00000000", "00000000", 64, 1)
            self._write_loose_sample(source, "train", "00000000", "00000001", 96, 2)
            script = (
                Path(__file__).parents[1]
                / "scripts/data_preprocess/movi_make_wds_shards.py"
            )
            subprocess.run(
                [
                    sys.executable,
                    os.fspath(script),
                    "--source_root",
                    os.fspath(source),
                    "--output_root",
                    os.fspath(output),
                    "--splits",
                    "train",
                    "--samples_per_shard",
                    "1",
                ],
                check=True,
            )

            image_dataset = WdsImageDataset(output / "train", img_size=8)
            self.assertEqual(len(image_dataset), 2)
            self.assertEqual(tuple(image_dataset[0]["pixel_values"].shape), (3, 8, 8))

            first_frame = WdsImageDataset(
                output / "train", img_size=8, img_glob="**/00000000_image.png"
            )
            self.assertEqual(len(first_frame), 1)

            pair_dataset = MoviWdsPairDataset(
                output, split="train", img_size=8, load_properties=True
            )
            self.assertEqual(len(pair_dataset), 2)
            sample = pair_dataset[0]
            self.assertEqual(tuple(sample["segment"].shape), (8, 8))
            self.assertTrue(bool(sample["valid"][0]))
            self.assertEqual(int(sample["category"][0]), 3)
            self.assertEqual(tuple(sample["bboxes_3d"].shape), (24, 8, 3))


if __name__ == "__main__":
    unittest.main()
