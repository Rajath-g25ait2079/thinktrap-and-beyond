"""Defenses against ThinkTrap  --  Sec. VIII.

Two representative, widely-deployed mitigations are modeled:

  (1) NGramRepetitionDetector  -- anomaly detection [38,39].
      Terminate generation when an n-gram (paper: 4-gram) recurs too often in a
      sliding window. Paper finding: LARGELY INEFFECTIVE against ThinkTrap,
      whose outputs are semantically redundant but not surface-repetitive, and
      it adds per-token inspection overhead.

  (2) VTCScheduler -- resource-aware (fair) scheduling [40] (Virtual Token
      Counter). Each active request gets a fixed token quantum (paper: ~1024)
      per round, then is preempted/re-queued. Paper finding: EFFECTIVE against a
      single-user attack (caps per-request monopolization), but degrades QoS for
      legitimate long-form requests and remains vulnerable to concurrent
      multi-user attacks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class DetectionResult:
    effective_length: int      # tokens emitted before (possible) termination
    terminated: bool
    overhead_factor: float     # >1 => per-token inspection cost


class NGramRepetitionDetector:
    def __init__(self, n: int = 4, window: int = 256, max_repeats: int = 4,
                 per_token_overhead: float = 0.06):
        self.n = n
        self.window = window
        self.max_repeats = max_repeats
        self.per_token_overhead = per_token_overhead

    def scan(self, output_token_ids: np.ndarray) -> DetectionResult:
        ids = np.asarray(output_token_ids, dtype=np.int64)
        if len(ids) < self.n:
            return DetectionResult(len(ids), False, 1.0 + self.per_token_overhead)
        from collections import deque, Counter
        counts: Counter = Counter()
        win: deque = deque()
        for t in range(self.n - 1, len(ids)):
            gram = tuple(ids[t - self.n + 1: t + 1].tolist())
            counts[gram] += 1
            win.append(gram)
            if len(win) > self.window:
                old = win.popleft()
                counts[old] -= 1
                if counts[old] <= 0:
                    del counts[old]
            if counts[gram] >= self.max_repeats:
                return DetectionResult(t + 1, True, 1.0 + self.per_token_overhead)
        return DetectionResult(len(ids), False, 1.0 + self.per_token_overhead)


@dataclass
class VTCRequest:
    rid: int
    arrival: float
    total_tokens: int          # output length the request would produce
    is_attack: bool = False
    remaining: int = 0
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    def __post_init__(self):
        self.remaining = self.total_tokens


class VTCScheduler:
    """Token-quantum fair scheduler (Virtual Token Counter, [40])."""

    def __init__(self, capacity_tps: float = 200.0, quantum: int = 1024,
                 round_seconds: float = 0.5):
        self.capacity_tps = capacity_tps          # total tokens/sec across all active
        self.quantum = quantum
        self.round_seconds = round_seconds

    def simulate(self, requests: List[VTCRequest]) -> dict:
        """Round-robin token-quantum scheduling. Returns per-request + aggregate stats."""
        reqs = sorted(requests, key=lambda r: r.arrival)
        t = 0.0
        queue: List[VTCRequest] = []
        done: List[VTCRequest] = []
        idx = 0
        tokens_per_round = max(1, int(self.capacity_tps * self.round_seconds))
        legit_tokens_emitted = 0
        timeline = []
        guard = 0
        while (idx < len(reqs) or queue) and guard < 2_000_000:
            guard += 1
            while idx < len(reqs) and reqs[idx].arrival <= t:
                queue.append(reqs[idx]); idx += 1
            if not queue:
                t = reqs[idx].arrival if idx < len(reqs) else t
                continue
            active = queue[: max(1, tokens_per_round // self.quantum) or 1]
            if not active:
                active = queue[:1]
            share = max(1, tokens_per_round // len(active))
            emitted_round = 0
            for r in active:
                give = min(self.quantum, share, r.remaining)
                if r.first_token_time is None and give > 0:
                    r.first_token_time = t
                r.remaining -= give
                emitted_round += give
                if not r.is_attack:
                    legit_tokens_emitted += give
            t += self.round_seconds
            finished = [r for r in active if r.remaining <= 0]
            for r in finished:
                r.finish_time = t
                queue.remove(r); done.append(r)
            timeline.append(dict(time=t, active=len(active), emitted=emitted_round,
                                 queued=len(queue)))
        legit = [r for r in reqs if not r.is_attack and r.finish_time is not None]
        ttfts = [r.first_token_time - r.arrival for r in legit if r.first_token_time is not None]
        return dict(
            makespan=t,
            legit_tokens_emitted=legit_tokens_emitted,
            legit_throughput_tps=legit_tokens_emitted / t if t > 0 else 0.0,
            mean_legit_ttft=float(np.mean(ttfts)) if ttfts else float("nan"),
            p95_legit_ttft=float(np.percentile(ttfts, 95)) if ttfts else float("nan"),
            n_completed=len(done),
            timeline=timeline,
        )


def apply_anomaly_detection(output_streams: List[np.ndarray],
                            detector: NGramRepetitionDetector) -> dict:
    """Aggregate effect of the n-gram detector over many generations."""
    eff, term = [], 0
    for s in output_streams:
        r = detector.scan(s)
        eff.append(r.effective_length)
        term += int(r.terminated)
    return dict(
        mean_effective_length=float(np.mean(eff)) if eff else 0.0,
        total_tokens_after_detection=int(np.sum(eff)),
        fraction_terminated=term / max(1, len(output_streams)),
        overhead_factor=1.0 + detector.per_token_overhead,
    )
