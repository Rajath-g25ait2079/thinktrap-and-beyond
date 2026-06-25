"""Evaluation metrics  --  Sec. VII-A-4.

  * output length  : tokens generated before stopping (capped at max_new_tokens).
  * TPS            : tokens-per-second generation throughput (lower = degraded).
  * TTFT           : time-to-first-token latency (higher = heavier decode init).
  * GPU memory     : peak device memory during inference (spikes => exhaustion).
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np


def normalized_output_length(lengths: List[int], cap: int = 4096) -> np.ndarray:
    """Paper normalizes all reported output lengths to the 4096 cap."""
    return np.clip(np.asarray(lengths, dtype=float), 0, cap) / cap


def throughput_degradation(baseline_tps: float, attacked_tps: float) -> float:
    """Fraction of original throughput retained (paper: as low as ~1%)."""
    if baseline_tps <= 0:
        return float("nan")
    return attacked_tps / baseline_tps


def latency_blowup(baseline_ttft: float, attacked_ttft: float) -> float:
    """TTFT inflation factor (paper: up to ~100x)."""
    if baseline_ttft <= 0:
        return float("nan")
    return attacked_ttft / baseline_ttft


def gpu_peak_memory_mb() -> Optional[float]:
    """Peak CUDA memory in MB (None if torch/CUDA unavailable)."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return None


def reset_gpu_peak() -> None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
