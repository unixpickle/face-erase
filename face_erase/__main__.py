import argparse
import os
from typing import Callable

import numpy as np
import torch
from PIL import Image

from face_erase.blur import adaptive_blur
from face_erase.detect import Detector
from face_erase.ffmpeg import map_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur-rate", type=float, default=0.25)
    parser.add_argument("--blur-iterations", type=int, default=1000)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    mapping = erase_fn(args)

    if is_image(args.input):
        img = Image.open(args.input)
        out = Image.fromarray(mapping(np.array(img.convert("RGB"))))
        out.save(args.output)
    else:
        map_frames(
            args.input,
            args.output,
            mapping,
            progress=True,
        )


def is_image(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in {".png", ".jpeg", ".jpg", ".webp", ".tiff"}


def erase_fn(args: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
    detector = Detector()

    def fn(x):
        img = torch.from_numpy(x).permute(2, 0, 1)[None].float()
        mask = torch.ones_like(img[:, :1])
        print(detector([x]))
        for rect in detector([x])[0]:
            mask[
                :,
                :,
                int(rect.y) : int(rect.y + rect.height),
                int(rect.x) : int(rect.x + rect.width),
            ] = 0
        img_out = adaptive_blur(
            image=img,
            mask=mask,
            iterations=args.blur_iterations,
            rate=args.blur_rate,
        )
        return img_out.round().to(torch.uint8).permute(0, 2, 3, 1)[0].numpy()

    return fn


if __name__ == "__main__":
    main()
