import os
import sys
import gc
from pathlib import Path
from typing import Any, Callable, TypeVar

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


T = TypeVar("T")


def hf_or_skip(description: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if _is_hf_access_error(exc):
            pytest.skip(f"{description} is unavailable from Hugging Face: {exc}")
        raise


def _is_hf_access_error(exc: Exception) -> bool:
    exc_type = type(exc)
    module = exc_type.__module__
    if module.startswith(("huggingface_hub", "requests", "urllib3")):
        return True
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return True

    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in (
            "401",
            "403",
            "can't load",
            "couldn't connect",
            "connection error",
            "gated repo",
            "huggingface.co",
            "local_files_only",
            "name or service not known",
            "offline",
            "repo not found",
            "unauthorized",
            "xethub",
        )
    )


def tl_parity_device() -> str:
    device = os.environ.get("EAP_TL_PARITY_DEVICE", "cpu").strip().lower()
    if not device:
        device = "cpu"
    if device == "gpu":
        device = "cuda"
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            pytest.skip(f"EAP_TL_PARITY_DEVICE={device} requested but CUDA is unavailable.")
    return device


@pytest.fixture(autouse=True)
def cleanup_torch_state():
    yield
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
