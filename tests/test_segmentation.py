"""Closed-form sanity checks for src/metrics/segmentation.py.

Run with: `uv run python -m unittest tests/test_segmentation.py`
"""

import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.segmentation import (
    per_image_fg_ari,
    per_image_mbo,
    per_image_miou,
)


class SegmentationMetrics(unittest.TestCase):
    def test_perfect_match(self):
        # gt and pred identical -> all metrics = 1 (ARI undefined on bg only).
        gt = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 0, 0],
                [2, 2, 0, 0],
            ],
            dtype=np.int64,
        )
        pred = gt.copy()
        self.assertAlmostEqual(per_image_fg_ari(gt, pred), 1.0)
        self.assertAlmostEqual(per_image_mbo(gt, pred), 1.0)
        # mIoU includes background (id 0) since ignore_background=False.
        self.assertAlmostEqual(per_image_miou(gt, pred), 1.0)

    def test_relabel_invariant(self):
        # Permuting predicted label ids must not change FG-ARI / mBO / mIoU.
        gt = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 0, 0],
                [2, 2, 0, 0],
            ],
            dtype=np.int64,
        )
        # Same partition as gt but with labels swapped (1<->2, 0 kept).
        pred = np.array(
            [
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=np.int64,
        )
        self.assertAlmostEqual(per_image_fg_ari(gt, pred), 1.0)
        self.assertAlmostEqual(per_image_mbo(gt, pred), 1.0)
        self.assertAlmostEqual(per_image_miou(gt, pred), 1.0)

    def test_partial_overlap(self):
        # gt has two 4-pixel instances; pred merges them into one 8-pixel blob.
        gt = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 0, 0],
                [2, 2, 0, 0],
            ],
            dtype=np.int64,
        )
        pred = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=np.int64,
        )
        # mBO: per-GT-object greedy max IoU. Both GT instances see the merged
        # pred 1: intersection 4, union 8 -> IoU = 0.5 each. Mean = 0.5.
        self.assertAlmostEqual(per_image_mbo(gt, pred), 0.5)
        # mIoU: pred ids present are {0, 1}; gt ids are {0, 1, 2}.
        #   gt 0 vs pred 0: inter=8, union=8 -> IoU 1.0
        #   gt 1 vs pred 1: inter=4, union=8 -> IoU 0.5
        #   gt 2 vs pred 1: inter=4, union=8 -> IoU 0.5  (gt 2 vs pred 0: 0)
        # Hungarian best assignment: (0->0, 1->1) sum=1.5, or (0->0, 2->1)
        # sum=1.5; either way matched sum = 1.5 over 3 gt ids -> 0.5.
        self.assertAlmostEqual(per_image_miou(gt, pred), 0.5)

    def test_no_foreground_returns_none(self):
        gt = np.zeros((4, 4), dtype=np.int64)
        pred = np.zeros((4, 4), dtype=np.int64)
        self.assertIsNone(per_image_fg_ari(gt, pred))
        self.assertIsNone(per_image_mbo(gt, pred))

    def test_ignore_background_in_miou(self):
        gt = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int64,
        )
        pred = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int64,
        )
        # Excluding background id 0: only gt 1 vs pred {0,1}; best = 1.0.
        self.assertAlmostEqual(per_image_miou(gt, pred, ignore_background=True), 1.0)


if __name__ == "__main__":
    unittest.main()
