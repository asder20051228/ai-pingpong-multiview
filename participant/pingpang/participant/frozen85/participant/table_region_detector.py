"""Small per-frame table-region gate using the trained ResNet18 ONNX model."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class TableRegionDetector:
    width, height = 320, 192

    def __init__(self, infer_fn):
        self.infer = infer_fn

    def box(self, frame_rgb: np.ndarray, orig_w: int, orig_h: int):
        image = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        # TableSegNet.forward normalizes inside the exported ONNX graph.
        # Supply RGB values in [0, 1] so normalization happens exactly once.
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        probability = self.infer(tensor)[0, 0]
        mask = probability > 0.5
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        area_fraction = float(mask.mean())
        if area_fraction < 0.002 or area_fraction > 0.4:
            return None
        # The exported segmentation head emits a 1/4-resolution mask (80x48
        # for a 320x192 input), so scale from the mask grid, not input pixels.
        mask_h, mask_w = mask.shape
        sx, sy = orig_w / mask_w, orig_h / mask_h
        return (float(xs.min() * sx), float(ys.min() * sy),
                float((xs.max() + 1) * sx), float((ys.max() + 1) * sy))
