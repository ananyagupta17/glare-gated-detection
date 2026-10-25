"""Learned daytime sun-glare restorer: a small U-Net trained on our paired v2 frames.

    restore(image: np.ndarray) -> np.ndarray      uint8 BGR in, same-shape uint8 BGR out

Trained by src/train_restore_daytime_net.py on v2 TRAIN pairs only (glared in, the paired
clean frame as target), so unlike Flare7K++ it has seen daytime sun glare, not nighttime
lens flare. It has only seen SYNTHETIC glare, which limits what it can claim on real glare.

DESIGN FOR SPEED

Glare and the camera's exposure response are smooth, low-frequency changes. The network
therefore runs on a half-resolution copy (624 x 192) and predicts a CORRECTION, which is
upsampled and added to the full-resolution frame. Fine detail comes straight from the
input; the network only has to model the veil and the exposure shift. About 0.5 M
parameters.

Weights: models/restore_daytime_net.pt (gitignored with the rest of models/).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = REPO_ROOT / "models" / "restore_daytime_net.pt"
NET_W, NET_H = 624, 192             # half of KITTI's 1242 x 375, rounded to multiples of 16


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class DayGlareUNet(nn.Module):
    """4-level U-Net predicting a residual correction for an RGB image in [0, 1]."""

    def __init__(self, base: int = 16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([block(3, c[0]), block(c[0], c[1]), block(c[1], c[2]),
                                  block(c[2], c[3])])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2)
                                 for i in range(3)])
        self.dec = nn.ModuleList([block(c[i] * 2, c[i]) for i in range(3)])
        self.head = nn.Conv2d(c[0], 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for i, enc in enumerate(self.enc):
            h = enc(h)
            if i < 3:
                skips.append(h)
                h = self.pool(h)
        for i in (2, 1, 0):
            h = self.up[i](h)
            h = self.dec[i](torch.cat([h, skips[i]], 1))
        return self.head(h)                      # correction, added to the input


class DaytimeNetRestorer:
    def __init__(self, weights: Path = WEIGHTS, device: str | None = None):
        if not Path(weights).is_file():
            raise FileNotFoundError(f"{weights} not found; run src/train_restore_daytime_net.py")
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model = DayGlareUNet(state.get("base", 16))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def restore(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an H x W x 3 uint8 array")
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        small = cv2.resize(rgb, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        corr = self.model(x)[0].float().cpu().numpy().transpose(1, 2, 0)
        corr = cv2.resize(corr, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.clip(rgb + corr, 0.0, 1.0)
        return cv2.cvtColor(np.rint(out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


_DEFAULT: DaytimeNetRestorer | None = None
_LOCK = threading.Lock()


def get_restorer() -> DaytimeNetRestorer:
    global _DEFAULT
    if _DEFAULT is None:
        with _LOCK:
            if _DEFAULT is None:
                _DEFAULT = DaytimeNetRestorer(device=os.getenv("DAYTIME_NET_DEVICE"))
    return _DEFAULT


def restore(image: np.ndarray) -> np.ndarray:
    return get_restorer().restore(image)
