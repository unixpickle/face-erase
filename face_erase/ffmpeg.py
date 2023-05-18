"""
Read and write frames from videos.
"""

import io
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
from tqdm.auto import tqdm

from .child_stream import ChildStream


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Optional[float]

    def with_fps(self, fps: float) -> "VideoInfo":
        return VideoInfo(
            width=self.width,
            height=self.height,
            fps=fps,
        )


def map_frames(
    input_path: str,
    output_path: str,
    fn: Callable[[np.ndarray], np.ndarray],
    fallback_fps: float = 24.0,
    progress: bool = False,
):
    """
    Modify a video by applying the given function to each frame.
    """
    info = video_info(input_path)
    frames = read_frames(
        input_path, info=info, force_fps=None if info.fps else fallback_fps
    )
    it = (fn(x) for x in frames)
    if progress:
        it = tqdm(it)
    write_frames(
        output_path=output_path,
        audio_input_path=input_path,
        info=info if info.fps else info.with_fps(fallback_fps),
        frames=it,
    )


def read_frames(
    path: str, info: Optional[VideoInfo] = None, force_fps: Optional[float] = None
) -> Iterator[np.ndarray]:
    """
    Read image frames from a video file using ffmpeg.
    """
    # Based on https://github.com/unixpickle/ffmpego/blob/6d92dd74560e18945db517a6b259ede1f2198391/video_reader.go
    if info is None:
        info = video_info(path)

    with ChildStream.create() as stream:
        with run_silent(
            [
                "ffmpeg",
                "-i",
                path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                *([] if not force_fps else ["-filter:v", f"fps=fps={force_fps:f}"]),
                stream.resource_url(),
            ],
            pass_fds=stream.pass_fds(),
        ):
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
    with ChildStream.create() as stream:
        with run_silent(
            [
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
            ],
            pass_fds=stream.pass_fds(),
        ):
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


@contextmanager
def run_silent(*args, **kwargs) -> Iterator[subprocess.Popen]:
    proc = subprocess.Popen(
        *args,
        **kwargs,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
        proc.wait()
        proc = None
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()


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
    return VideoInfo(width=width, height=height, fps=fps)
