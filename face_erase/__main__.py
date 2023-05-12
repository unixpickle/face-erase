import argparse

import numpy as np
from PIL import Image

from face_erase.detect import Detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    img = Image.open(args.input)
    detector = Detector()
    rects = detector([img])[0]

    out = np.array(img.convert("RGB"))
    for rect in rects:
        print(rect)
        out[
            int(rect.y) : int(rect.y + rect.height),
            int(rect.x) : int(rect.x + rect.width),
        ] = 0

    Image.fromarray(out).save(args.output)


if __name__ == "__main__":
    main()
