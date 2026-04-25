from __future__ import annotations


class ProcessingCancelledError(RuntimeError):
    def __init__(self, message: str = "Processing was canceled.") -> None:
        super().__init__(message)
