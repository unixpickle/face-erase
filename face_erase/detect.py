"""
Detect faces in images.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from face_erase.ultralytics_util import non_max_suppression

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "binary",
    "yolov8n-face-640x640.torchscript",
)


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float


class Detector:
    """
    Find face bounding boxes in images using a pre-trained ultralytics model.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Union[torch.device, str] = "cpu",
        resolution: int = 640,
        stride: int = 640,  # prevent any non-square inputs
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.resolution = resolution
        self.stride = stride

    def __call__(
        self,
        imgs: Iterable[Union[torch.Tensor, np.ndarray, Image.Image]],
        confidence: float = 0.01,
        iou: float = 0.5,
    ) -> List[List[Rect]]:
        results = []
        for img in imgs:
            img, unscale = self._preprocess(img)
            pred = self.model(img[None])
            results.extend(
                self._postprocess(pred, unscale, confidence=confidence, iou=iou)
            )
        return results

    def _preprocess(
        self, img: Union[torch.Tensor, np.ndarray, Image.Image]
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(self.device)
        elif not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.array(img.convert("RGB"))).to(self.device)

        img = img.float() / 255
        img = img.permute(2, 0, 1)

        ch, height, width = img.shape
        assert ch == 3, f"unexpected image shape {img.shape=}"
        scale = self.resolution / max(height, width)
        new_height = round(height * scale)
        new_width = round(width * scale)
        pad_height = self._padding(new_height)
        pad_width = self._padding(new_width)
        result = torch.zeros(
            (3, new_height + pad_height, new_width + pad_width),
            device=self.device,
            dtype=torch.float32,
        )
        result[
            :,
            pad_height // 2 : pad_height // 2 + new_height,
            pad_width // 2 : pad_width // 2 + new_width,
        ] = F.interpolate(img[None].float(), (new_height, new_width), mode="bilinear")[
            0
        ]

        def unscale_coords(coords: torch.Tensor) -> torch.Tensor:
            assert coords.shape[-1] == 4
            coords = coords.clone()
            coords = coords - torch.tensor([pad_width // 2, pad_height // 2] * 2).to(
                coords
            )
            return coords / scale

        return result, unscale_coords

    def _postprocess(
        self,
        preds: Tuple[torch.Tensor, Sequence[Any]],
        unscale: Callable[[torch.Tensor], torch.Tensor],
        confidence: float,
        iou: float,
    ) -> List[List[Rect]]:
        preds = non_max_suppression(
            preds, confidence, iou, agnostic=False, max_det=1000, classes=None, nc=1
        )
        return [
            [
                Rect(x=x, y=y, width=xw - x, height=yh - y)
                for x, y, xw, yh in unscale(pred[..., :4]).tolist()
            ]
            for pred in preds
        ]

    def _padding(self, size: int) -> int:
        return (self.stride - size % self.stride) % self.stride
