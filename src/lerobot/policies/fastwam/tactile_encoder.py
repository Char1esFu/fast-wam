"""
UniVTAC tactile encoder of the Fast-WAM tactile expert.

The UniVTAC ResNet (`encoder/network.py` `Tactile` of the UniVTAC repo) maps one tactile image in [0, 1] to one
latent: torchvision ResNet whose classifier head is the latent projection (`num_classes = latent_dim`).
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

SUPPORTED_BACKBONES = ("resnet18", "resnet34", "resnet50")


class UniVTACTactileEncoder(nn.Module):
    """
    Tactile images `[N, 3, H, W]` in [0, 1] to latents `[N, latent_dim]`, resized to `input_size` first. The
    BatchNorm statistics stay at their pretraining values (the layers stay in eval mode).

    Args:
        backbone: "resnet18", "resnet34" or "resnet50".
        latent_dim: Width of the latent, the `num_classes` of the ResNet.
        input_size: Height and width the images are resized to.
        pretrained_path: UniVTAC checkpoint (`best.pth`). A missing file only warns: the weights then come from
            the policy checkpoint.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        latent_dim: int = 512,
        input_size: tuple[int, int] = (224, 224),
        pretrained_path: str | None = None,
    ):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"tactile_backbone must be one of {SUPPORTED_BACKBONES}, got {backbone!r}")
        from torchvision import models

        self.backbone = getattr(models, backbone)(num_classes=latent_dim)
        self.input_size = tuple(input_size)
        if pretrained_path:
            path = Path(pretrained_path).expanduser()
            if path.exists():
                self.load_univtac_weights(path)
            else:
                logging.warning(f"UniVTAC tactile checkpoint {path} not found, keeping the initialization.")

    def load_univtac_weights(self, path: Path) -> None:
        """Load the backbone of a UniVTAC checkpoint (full `Tactile` state dict or bare ResNet state dict)."""
        state = torch.load(str(path), map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        state = {k.removeprefix("backbone."): v for k, v in state.items() if not k.startswith("decoders.")}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"UniVTAC checkpoint {path} misses backbone keys {missing[:5]}...")
        logging.info(f"Loaded the UniVTAC tactile encoder from {path} ({len(unexpected)} unused keys).")

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if tuple(images.shape[-2:]) != self.input_size:
            images = F.interpolate(images.float(), size=self.input_size, mode="bilinear", align_corners=False)
        return self.backbone(images.to(self.backbone.fc.weight.dtype))
