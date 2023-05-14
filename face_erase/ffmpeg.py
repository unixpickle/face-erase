"""
Read and write frames from videos.
"""

import io
import re
import subprocess
from typing import Iterator, Tuple

import numpy as np

from .child_stream import ChildStream


def read_frames(path: str) -> Iterator[np.ndarray]:
    width, height = video_resolution(path)

    stream = ChildStream.create()
    try:
        args = [
            "ffmpeg",
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            stream.resource_url(),
        ]
        proc = subprocess.Popen(
            args,
            pass_fds=stream.pass_fds(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            reader = stream.connect()
            frame_size = width * height * 3
            bufreader = io.BufferedReader(reader, buffer_size=frame_size)
            while True:
                buf = bufreader.read(frame_size)
                if not buf:
                    break
                yield np.frombuffer(buf, dtype=np.uint8).reshape([height, width, 3])
            proc.wait()
        except:
            proc.kill()
            proc.wait()
    finally:
        stream.close()


def video_resolution(path) -> Tuple[int, int]:
    """
    Get the width and height of a video.
    """
    # Based on https://github.com/unixpickle/ffmpego/blob/6d92dd74560e18945db517a6b259ede1f2198391/video_info.go
    proc = subprocess.Popen(
        ["ffmpeg", "-i", path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, out = proc.communicate()
    lines = out.decode("utf-8").splitlines()
    size_re = re.compile(r" ([0-9]+)x([0-9]+)(,| )")
    for line in lines:
        if "Video:" not in line:
            continue
        match = size_re.search(line)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    raise RuntimeError("could not infer size from ffmpeg output")
