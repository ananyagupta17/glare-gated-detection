"""Flare7K++ Uformer adapter for the glare-gated detection harness.

The public contract used by ``run_experiment.py`` is deliberately small:

    restore(image: np.ndarray) -> np.ndarray

Both input and output are uint8 BGR images with identical dimensions.  Model
loading is lazy, so importing this module does not allocate the network.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import types
import warnings
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "restoration" / "net_g_last.pth"


def find_flare7k_repo() -> Path:
    """Return the first conventional Flare7K checkout location.

    ``external/Flare7K`` keeps a fresh clone self-contained.  The sibling path
    remains supported for existing team workspaces.
    """
    candidates = (
        PROJECT_ROOT / "external" / "Flare7K",
        PROJECT_ROOT.parent / "Flare7K",
    )
    for candidate in candidates:
        if (candidate / "basicsr" / "archs" / "uformer_arch.py").is_file():
            return candidate
    return candidates[0]


DEFAULT_FLARE7K_REPO = find_flare7k_repo()


class Flare7KRestorer:
    """Run the pretrained Flare7K++ Uformer on OpenCV images."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        flare7k_repo: str | Path = DEFAULT_FLARE7K_REPO,
        device: str = "auto",
        tile_size: int = 512,
        overlap: int = 128,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.flare7k_repo = Path(flare7k_repo).resolve()
        self.tile_size = tile_size
        self.overlap = overlap
        if tile_size != 512:
            raise ValueError("This checkpoint was trained for 512 x 512 inputs")
        if not 0 <= overlap < tile_size:
            raise ValueError("overlap must be between 0 and tile_size - 1")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required. Install requirements.txt first.") from exc
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
        self.device = torch.device(device)

        started = perf_counter()
        self.model = self._load_model()
        self.load_seconds = perf_counter() - started

    def _load_model(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Flare7K checkpoint not found: {self.checkpoint}")
        arch_file = self.flare7k_repo / "basicsr" / "archs" / "uformer_arch.py"
        if not arch_file.is_file():
            raise FileNotFoundError(
                f"Flare7K repository not found at {self.flare7k_repo}. "
                "Expected basicsr/archs/uformer_arch.py. From this repository run: "
                "git clone https://github.com/ykdai/Flare7K.git external/Flare7K "
                "or set FLARE7K_REPO to an existing checkout."
            )

        # Loading the file directly avoids BasicSR importing every architecture,
        # model and dataset.  The Uformer file only needs the registry decorator,
        # so provide that tiny interface without executing basicsr/__init__.py.
        class _Registry:
            def __init__(self):
                self.objects = {}

            def register(self):
                def decorator(item):
                    self.objects[item.__name__] = item
                    return item
                return decorator

        basicsr_module = types.ModuleType("basicsr")
        basicsr_module.__path__ = [str(self.flare7k_repo / "basicsr")]
        utils_module = types.ModuleType("basicsr.utils")
        utils_module.__path__ = [str(self.flare7k_repo / "basicsr" / "utils")]
        registry_module = types.ModuleType("basicsr.utils.registry")
        registry_module.ARCH_REGISTRY = _Registry()
        module_names = ("basicsr", "basicsr.utils", "basicsr.utils.registry")
        previous_modules = {name: sys.modules.get(name) for name in module_names}
        sys.modules.update({
            "basicsr": basicsr_module,
            "basicsr.utils": utils_module,
            "basicsr.utils.registry": registry_module,
        })
        try:
            spec = importlib.util.spec_from_file_location("flare7k_uformer_arch", arch_file)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not import Uformer architecture from {arch_file}")
            module = importlib.util.module_from_spec(spec)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")
                warnings.filterwarnings("ignore", message=r"torch\.meshgrid:.*")
                spec.loader.exec_module(module)
                model = module.Uformer(img_size=512, img_ch=3, output_ch=6)
        except ModuleNotFoundError as exc:
            if exc.name in {"timm", "einops"}:
                raise RuntimeError(
                    f"Missing '{exc.name}'. Run: pip install timm einops"
                ) from exc
            raise
        finally:
            for name, previous in previous_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        state = self.torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "params_ema" in state:
            state = state["params_ema"]
        elif isinstance(state, dict) and "params" in state:
            state = state["params"]
        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint does not contain a PyTorch state dictionary")
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        return model

    @staticmethod
    def _positions(length: int, tile: int, stride: int) -> list[int]:
        if length <= tile:
            return [0]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] != length - tile:
            positions.append(length - tile)
        return positions

    def _predict_tile(self, rgb_float: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(rgb_float.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.torch.float32)
        with self.torch.inference_mode():
            output = self.model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            # Flare7K++ supervises the first three output channels as the clean image.
            clean = output[:, :3].clamp(0.0, 1.0)
        return clean[0].detach().cpu().numpy().transpose(1, 2, 0)

    def _predict_resized(self, rgb: np.ndarray) -> np.ndarray:
        """Predict the clean image at model scale using feathered 512px tiles."""
        height, width = rgb.shape[:2]
        scale = self.tile_size / min(height, width)
        scaled_w = max(self.tile_size, int(round(width * scale)))
        scaled_h = max(self.tile_size, int(round(height * scale)))
        resized = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        source = resized.astype(np.float32) / 255.0

        stride = self.tile_size - self.overlap
        xs = self._positions(scaled_w, self.tile_size, stride)
        ys = self._positions(scaled_h, self.tile_size, stride)
        total = np.zeros_like(source, dtype=np.float32)
        weights = np.zeros((scaled_h, scaled_w, 1), dtype=np.float32)

        if len(xs) == 1 and len(ys) == 1:
            blend = np.ones((self.tile_size, self.tile_size, 1), dtype=np.float32)
        else:
            ramp = np.hanning(self.tile_size).astype(np.float32)
            # Keep image borders represented while reducing seams in overlaps.
            blend = (0.05 + 0.95 * np.outer(ramp, ramp))[..., None]

        for y in ys:
            for x in xs:
                predicted = self._predict_tile(source[y:y + self.tile_size, x:x + self.tile_size])
                total[y:y + self.tile_size, x:x + self.tile_size] += predicted * blend
                weights[y:y + self.tile_size, x:x + self.tile_size] += blend
        return resized, np.clip(total / np.maximum(weights, 1e-6), 0.0, 1.0)

    def restore(self, image: np.ndarray) -> np.ndarray:
        """Restore one uint8 BGR image and preserve its original dimensions."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an H x W x 3 NumPy array")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_rgb, predicted_clean = self._predict_resized(original_rgb)

        # Match the official Flare7K test_large.py resolution-preserving method:
        # estimate flare at model scale, resize it, then subtract from the source.
        resized_float = resized_rgb.astype(np.float32) / 255.0
        predicted_flare = np.abs(resized_float - predicted_clean)
        flare_original = cv2.resize(
            predicted_flare,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        restored_rgb = np.abs(original_rgb.astype(np.float32) / 255.0 - flare_original)
        restored_rgb = np.clip(np.rint(restored_rgb * 255.0), 0, 255).astype(np.uint8)
        return cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)


_DEFAULT_RESTORER: Flare7KRestorer | None = None
_LOCK = threading.Lock()


def get_restorer() -> Flare7KRestorer:
    """Return the process-wide model, loading it only on first use."""
    global _DEFAULT_RESTORER
    if _DEFAULT_RESTORER is None:
        with _LOCK:
            if _DEFAULT_RESTORER is None:
                _DEFAULT_RESTORER = Flare7KRestorer(
                    checkpoint=os.getenv("FLARE7K_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
                    flare7k_repo=os.getenv("FLARE7K_REPO", str(DEFAULT_FLARE7K_REPO)),
                    device=os.getenv("FLARE7K_DEVICE", "auto"),
                )
    return _DEFAULT_RESTORER


def restore(image: np.ndarray) -> np.ndarray:
    """Harness entry point: uint8 BGR input -> same-sized uint8 BGR output."""
    return get_restorer().restore(image)
