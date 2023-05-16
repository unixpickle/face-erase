"""
Read and write frames from videos.
"""

import io
import re
import subprocess
from dataclasses import dataclass
from email.mime import audio
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from .child_stream import ChildStream


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float


def read_frames(path: str, info: Optional[VideoInfo] = None) -> Iterator[np.ndarray]:
    """
    Read image frames from a video file using ffmpeg.
    """
    # Based on https://github.com/unixpickle/ffmpego/blob/6d92dd74560e18945db517a6b259ede1f2198391/video_reader.go
    if info is None:
        info = video_info(path)

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
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            reader = stream.connect()
            frame_size = info.width * info.height * 3
            bufreader = io.BufferedReader(reader, buffer_size=frame_size)
            while True:
                buf = bufreader.read(frame_size)
                if not buf:
                    break
                yield np.frombuffer(buf, dtype=np.uint8).reshape(
                    [info.height, info.width, 3]
                )
            proc.wait()
        except:
            proc.kill()
            proc.wait()
            raise
    finally:
        stream.close()


def write_frames(
    output_path: str,
    audio_input_path: str,
    info: VideoInfo,
    frames: Iterable[np.ndarray],
):
    """
    Create a video file and write frames to it.

    Copies audio from an existing file.
    """
    # Based on https://github.com/unixpickle/ffmpego/blob/6d92dd74560e18945db517a6b259ede1f2198391/video_writer.go
    stream = ChildStream.create()
    try:
        args = [
            "ffmpeg",
            "-y",
            # Video format
            "-r",
            f"{info.fps:f}",
            "-s",
            f"{info.width}x{info.height}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-probesize",
            "32",
            "-thread_queue_size",
            "10000",
            "-i",
            stream.resource_url(),
            # Add audio source
            "-i",
            audio_input_path,
            "-c:a",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            # Output parameters
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            output_path,
        ]
        proc = subprocess.Popen(
            args,
            pass_fds=stream.pass_fds(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            writer = stream.connect()
            bufwriter = io.BufferedWriter(writer)
            for frame in frames:
                assert frame.shape == (
                    info.height,
                    info.width,
                    3,
                ), f"unexpected shape {frame.shape}"
                assert frame.dtype == np.uint8, f"unexpected dtype {frame.dtype}"
                bufwriter.write(frame.tobytes(order="C"))
            bufwriter.flush()
            stream.close()
            proc.wait()
        except:
            proc.kill()
            proc.wait()
            raise
    finally:
        stream.close()


def video_info(path: str) -> VideoInfo:
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
    fps_re = re.compile(r" ([0-9\\.]*) fps,")
    width, height = None, None
    fps = None
    for line in lines:
        if "Video:" not in line:
            continue
        match = size_re.search(line)
        if match is not None:
            width, height = int(match.group(1)), int(match.group(2))
        match = fps_re.search(line)
        if match is not None:
            fps = float(match.group(1))
    if width is None:
        raise RuntimeError("could not infer size from ffmpeg output")
    if fps is None:
        raise RuntimeError("could not infer fps from ffmpeg output")
    return VideoInfo(width=width, height=height, fps=fps)
