"""A small Grad-CAM implementation for ResNet-style image classifiers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    """Compute Grad-CAM for a chosen convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients: torch.Tensor) -> None:
        self.gradients = gradients

    def generate(
        self, image_tensor: torch.Tensor, target_class: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Return a normalized heatmap and the class used to create it."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        predicted_class = int(logits.argmax(dim=1).item())
        target_class = predicted_class if target_class is None else int(target_class)
        logits[0, target_class].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook did not receive activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = functional.relu(cam)
        cam = functional.interpolate(
            cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class

    def close(self) -> None:
        """Remove the PyTorch hook when Grad-CAM is no longer needed."""
        self._hook.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a Grad-CAM heatmap onto an original RGB PIL image."""
    image_array = np.asarray(image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0])))
    color_map = colormaps["jet"](heatmap)[..., :3] * 255
    blended = (1 - alpha) * image_array + alpha * color_map
    return blended.astype(np.uint8)

