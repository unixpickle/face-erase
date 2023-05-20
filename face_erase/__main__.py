import argparse
import os
from typing import Callable

import numpy as np
from PIL import Image

from face_erase.detect import Detector
from face_erase.ffmpeg import map_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    mapping = erase_fn()

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


def erase_fn() -> Callable[[np.ndarray], np.ndarray]:
    detector = Detector()

    def fn(x):
        out = np.array(x)
        for rect in detector([x])[0]:
            out[
                int(rect.y) : int(rect.y + rect.height),
                int(rect.x) : int(rect.x + rect.width),
            ] = 0
        return out

    return fn


if __name__ == "__main__":
    main()
