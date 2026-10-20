"""Flare7K++ restoration without visible tile seams. Same contract as src/restore.py:

    restore(image: np.ndarray) -> np.ndarray      uint8 BGR in, same-shape uint8 BGR out

Built on the restoration handover by Vihaan Bishnoi (src/restore.py, branch
codex/restoration-handover, commit 7f2f73a). Model loading, the checkpoint path
(models/restoration/net_g_last.pth, or FLARE7K_CHECKPOINT), the Flare7K source path and
the flare-subtraction step are all his, used unmodified by subclassing
Flare7KRestorer. Only the step that predicts the clean image at model scale is replaced.
Flare7K++ itself: Dai et al., S-Lab License 1.0 (THIRD_PARTY_LICENSES, external/Flare7K).

WHY

The handover tiles the frame (shorter side resized to 512) into 512 x 512 tiles with a
128 px overlap. Neighbouring tiles remove different amounts of glare, and the blend
cannot hide it: restored frames show hard vertical edges at the tile boundaries (e.g.
x = 843, 867, 937 on KITTI val 003507; see results/restoration_samples).

TWO MODES

  overlap   the handover's tiling with a larger overlap (default 384, so a 128 px
            stride). Each column is covered by more tiles, so a disagreement between
            two tiles is averaged over more of them.
  whole     one forward pass over the whole frame, no tiles. The Uformer reshapes its
            tokens as sqrt(L) x sqrt(L), so it only accepts SQUARE inputs (a 512 x 1696
            frame fails). The frame is therefore letterboxed: resized so the long side
            is 512 (1242 x 375 -> 512 x 155), reflect-padded to 512 x 512, predicted,
            and cropped back. Only the glare layer is estimated at that scale; it is
            upsampled and subtracted from the full-resolution original, exactly as the
            handover does, so scene detail comes from the original frame.

Select the mode with RESTORE_SEAMLESS_MODE (overlap | whole, default whole) and the
overlap with RESTORE_SEAMLESS_OVERLAP.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restore import (DEFAULT_CHECKPOINT, DEFAULT_FLARE7K_REPO,  # noqa: E402
                     Flare7KRestorer)

MODES = ("overlap", "whole")


class SeamlessFlare7KRestorer(Flare7KRestorer):
    def __init__(self, mode: str = "whole", overlap: int = 384, **kwargs) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        super().__init__(overlap=overlap, **kwargs)   # overlap is validated by the parent

    def _predict_resized(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "overlap":
            return super()._predict_resized(rgb)      # same tiling, larger overlap
        return self._predict_whole(rgb)

    def _predict_whole(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Letterbox to one 512 x 512 input. Returns (resized frame, predicted clean)
        at the letterboxed scale, the same pair the parent's restore() expects."""
        height, width = rgb.shape[:2]
        size = self.tile_size
        scale = size / max(height, width)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        source = resized.astype(np.float32) / 255.0

        pad_h, pad_w = size - new_h, size - new_w
        top, left = pad_h // 2, pad_w // 2
        canvas = cv2.copyMakeBorder(source, top, pad_h - top, left, pad_w - left,
                                    cv2.BORDER_REFLECT_101)
        predicted = self._predict_tile(canvas)
        return resized, predicted[top:top + new_h, left:left + new_w]


_DEFAULT: SeamlessFlare7KRestorer | None = None
_LOCK = threading.Lock()


def get_restorer() -> SeamlessFlare7KRestorer:
    global _DEFAULT
    if _DEFAULT is None:
        with _LOCK:
            if _DEFAULT is None:
                _DEFAULT = SeamlessFlare7KRestorer(
                    mode=os.getenv("RESTORE_SEAMLESS_MODE", "whole"),
                    overlap=int(os.getenv("RESTORE_SEAMLESS_OVERLAP", "384")),
                    checkpoint=os.getenv("FLARE7K_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
                    flare7k_repo=os.getenv("FLARE7K_REPO", str(DEFAULT_FLARE7K_REPO)),
                    device=os.getenv("FLARE7K_DEVICE", "auto"),
                )
    return _DEFAULT


def restore(image: np.ndarray) -> np.ndarray:
    """Harness entry point: uint8 BGR input -> same-sized uint8 BGR output."""
    return get_restorer().restore(image)
