"""Device resolution shared by training and evaluation.

Kept torch-free on purpose: ``resolve_device`` takes CUDA availability as a
plain boolean so it is trivially testable by monkeypatching
``torch.cuda.is_available`` at the call site (or by passing the flag
directly) — no GPU and no torch import needed in tests.
"""

from __future__ import annotations

VALID_DEVICES = ("auto", "cpu", "cuda")


def resolve_device(requested: str, cuda_available: bool) -> str:
    """Resolve an ``auto | cpu | cuda`` setting against real CUDA availability.

    - ``auto`` → ``cuda`` when available, else ``cpu``.
    - ``cuda`` → ``cuda``, or a clear error when CUDA is unavailable
      (callers map it to exit code 2).

    Pure: no imports beyond stdlib, so the base env can test it.
    """
    if requested not in VALID_DEVICES:
        raise ValueError(f"unknown device '{requested}' (expected {' | '.join(VALID_DEVICES)})")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "device 'cuda' requested but CUDA is not available on this machine "
                "(checked via torch.cuda.is_available)"
            )
        return "cuda"
    return "cuda" if cuda_available else "cpu"
