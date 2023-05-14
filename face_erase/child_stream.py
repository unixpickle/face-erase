"""
Communicate with ffmpeg subprocesses.

Ported from https://github.com/unixpickle/ffmpego/blob/6d92dd74560e18945db517a6b259ede1f2198391/child_stream.go#L57
"""

import socket
from abc import ABC, abstractmethod
from typing import BinaryIO, List


class ChildStream(ABC):
    @abstractmethod
    def pass_fds(self) -> List:
        pass

    @abstractmethod
    def resource_url(self) -> str:
        pass

    @abstractmethod
    def connect(self) -> BinaryIO:
        pass

    @abstractmethod
    def close(self):
        pass

    @staticmethod
    def create() -> "ChildStream":
        # TODO: support pipes as well
        return ChildSocketStream()


class ChildSocketStream(ChildStream):
    def __init__(self):
        super().__init__()
        self.sock = socket.socket()
        self.sock.bind(("localhost", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self._conn = None

    def pass_fds(self) -> List:
        return []

    def resource_url(self) -> str:
        return f"tcp://localhost:{self.port}"

    def connect(self) -> BinaryIO:
        conn, _ = self.sock.accept()
        self._conn = conn
        return self._conn.makefile("rwb")

    def close(self):
        if self._conn is not None:
            self._conn.close()
        self.sock.close()
