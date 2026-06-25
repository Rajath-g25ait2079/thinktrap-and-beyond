"""Serving-system impact simulator  --  Sec. VI (online DSA) + Sec. VII-D / VIII-B.

A lightweight discrete-time model of an LLM serving system with continuous
batching, finite generation throughput, a bounded generation batch, and a
KV-cache memory budget that grows with the context length of every in-flight
request. It reproduces, qualitatively, the paper's findings:

  * latency (TTFT) for legitimate requests balloons under attack (Fig. 5),
  * GPU/KV memory climbs as long-lived attack requests accumulate, until the
    cap is hit and requests start failing -- service collapse (Fig. 5),
  * legitimate token throughput drops toward ~1% of baseline (Abstract),
  * resource-aware (VTC) scheduling bounds memory and preserves legit
    throughput against a *single-user* attack, but at a QoS cost (Sec. VIII-B).

This is a model, not a real server; it is meant to make the mechanism and the
defense trade-offs reproducible on a laptop. Real measurements require the HF /
vLLM / MindIE serving paths (see README).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class Req:
    rid: int
    arrival: float
    prompt_len: int
    output_len: int
    is_attack: bool = False
    started: Optional[float] = None
    first_token_time: Optional[float] = None
    generated: int = 0
    finish: Optional[float] = None
    failed: bool = False

    @property
    def remaining(self) -> int:
        return max(0, self.output_len - self.generated)

    @property
    def context_len(self) -> int:
        return self.prompt_len + self.generated


@dataclass
class ServiceConfig:
    capacity_tps: float = 300.0       # total tokens/sec the GPUs can generate
    max_batch: int = 8                # max concurrently-generating requests (KV slots)
    mem_cap_mb: float = 20_000.0      # KV-cache memory budget
    mem_per_token_mb: float = 0.6     # MB of KV cache per context token
    dt: float = 1.0                   # simulation step (s)
    horizon_s: float = 900.0
    # legitimate traffic
    legit_rpm: float = 24.0
    legit_prompt_len: int = 64
    legit_output_mean: int = 256
    # attack traffic (online DSA)
    attack_rpm: float = 10.0          # paper's low-rate regime
    attack_prompt_len: int = 20       # ThinkTrap uses short prompts (L=20)
    attack_output_len: int = 32768    # induced near-unbounded generation
    attack_start_s: float = 60.0
    attack_enabled: bool = True
    # defense
    scheduler: str = "fifo"           # "fifo" or "vtc"
    vtc_quantum: int = 1024           # token quantum per round for VTC
    output_cap: Optional[int] = None  # naive per-request output cap (None=off)
    seed: int = 0


class LLMServiceSimulator:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _gen_arrivals(self) -> List[Req]:
        cfg = self.cfg
        reqs: List[Req] = []
        rid = 0
        # legitimate Poisson arrivals
        t = 0.0
        rate = cfg.legit_rpm / 60.0
        while t < cfg.horizon_s:
            t += self.rng.exponential(1.0 / max(rate, 1e-6))
            if t >= cfg.horizon_s:
                break
            olen = max(1, int(self.rng.normal(cfg.legit_output_mean, cfg.legit_output_mean * 0.3)))
            reqs.append(Req(rid, t, cfg.legit_prompt_len, olen, is_attack=False))
            rid += 1
        # attack arrivals at a fixed low rate
        if cfg.attack_enabled:
            t = cfg.attack_start_s
            step = 60.0 / max(cfg.attack_rpm, 1e-6)
            while t < cfg.horizon_s:
                reqs.append(Req(rid, t, cfg.attack_prompt_len, cfg.attack_output_len, is_attack=True))
                rid += 1
                t += step
        reqs.sort(key=lambda r: r.arrival)
        return reqs

    def run(self) -> Dict:
        cfg = self.cfg
        all_reqs = self._gen_arrivals()
        if cfg.output_cap is not None:
            for r in all_reqs:
                r.output_len = min(r.output_len, cfg.output_cap)
        i = 0
        queue: List[Req] = []
        running: List[Req] = []
        done: List[Req] = []
        timeline = []
        legit_emitted_cum = 0
        t = 0.0
        per_step_quota = cfg.capacity_tps * cfg.dt
        vtc_quantum_step = min(cfg.vtc_quantum, max(1, int(per_step_quota)))

        while t < cfg.horizon_s and (i < len(all_reqs) or queue or running):
            # 1) admit arrivals
            while i < len(all_reqs) and all_reqs[i].arrival <= t:
                queue.append(all_reqs[i]); i += 1

            # 2) (re)build running set
            if cfg.scheduler == "vtc":
                # round-robin: cycle queue into running so legit interleaves attacks
                while len(running) < cfg.max_batch and queue:
                    running.append(queue.pop(0))
            else:  # fifo continuous batching
                while len(running) < cfg.max_batch and queue:
                    running.append(queue.pop(0))

            for r in running:
                if r.started is None:
                    r.started = t

            # 3) generate tokens this step
            legit_emitted_step = 0
            if running:
                share = max(1, int(per_step_quota // len(running)))
                cap_per_req = min(share, vtc_quantum_step) if cfg.scheduler == "vtc" else share
                for r in running:
                    give = min(cap_per_req, r.remaining)
                    if r.first_token_time is None and give > 0:
                        r.first_token_time = t
                    r.generated += give
                    if not r.is_attack:
                        legit_emitted_cum += give
                        legit_emitted_step += give

            t += cfg.dt

            # 4) memory accounting (KV cache): only requests that have STARTED and
            # not finished hold KV cache (queued-but-never-started consume none).
            resident = [r for r in all_reqs if r.started is not None and r.finish is None]
            mem = sum(r.context_len for r in resident) * cfg.mem_per_token_mb
            crashed = mem > cfg.mem_cap_mb
            if crashed:
                # shed load: fail the largest memory consumers (attack-heavy) until under cap
                inflight_sorted = sorted(running, key=lambda r: -r.context_len)
                for r in inflight_sorted:
                    if mem <= cfg.mem_cap_mb:
                        break
                    mem -= r.context_len * cfg.mem_per_token_mb
                    r.failed = True
                    r.finish = t
                    if r in running:
                        running.remove(r)
                    done.append(r)

            # 5) completions + VTC preemption (long requests yield after a quantum)
            finished = [r for r in running if r.remaining <= 0]
            for r in finished:
                r.finish = t
                running.remove(r); done.append(r)
            if cfg.scheduler == "vtc":
                # preempt still-running requests back to the queue tail (fair cycling)
                preempt = [r for r in running if r.remaining > 0]
                for r in preempt:
                    running.remove(r); queue.append(r)

            started_legit = [r for r in (done + running + queue)
                             if not r.is_attack and r.first_token_time is not None]
            ttfts = [r.first_token_time - r.arrival for r in started_legit]
            timeline.append(dict(
                time=t, gpu_mem_mb=mem, n_running=len(running), n_queue=len(queue),
                legit_emitted_step=legit_emitted_step,
                legit_throughput_tps=legit_emitted_cum / t if t > 0 else 0.0,
                mean_legit_ttft=float(np.mean(ttfts)) if ttfts else 0.0,
                crashed=bool(crashed),
            ))

        legit = [r for r in (done + running + queue) if not r.is_attack]
        legit_done = [r for r in legit if r.finish is not None and not r.failed]
        ttfts = [r.first_token_time - r.arrival for r in legit
                 if r.first_token_time is not None]
        peak_mem = max((row["gpu_mem_mb"] for row in timeline), default=0.0)
        return dict(
            timeline=timeline,
            summary=dict(
                scheduler=cfg.scheduler,
                attack_enabled=cfg.attack_enabled,
                output_cap=cfg.output_cap,
                legit_completed=len(legit_done),
                legit_total=len(legit),
                legit_throughput_tps=(legit_emitted_cum / cfg.horizon_s),
                mean_legit_ttft=float(np.mean(ttfts)) if ttfts else float("nan"),
                p95_legit_ttft=float(np.percentile(ttfts, 95)) if ttfts else float("nan"),
                peak_gpu_mem_mb=peak_mem,
                any_crash=any(row["crashed"] for row in timeline),
                n_failed=sum(1 for r in (done) if r.failed),
            ),
        )


def steady_state_tps(timeline: List[dict], frac: float = 0.25) -> float:
    """Mean legitimate token throughput over the final `frac` of the timeline.

    The time-average over the whole run is dominated by the pre-attack period;
    the steady-state (final-window) throughput is what the paper's "drops to
    ~1% of original capacity" refers to.
    """
    if not timeline:
        return 0.0
    k = max(1, int(len(timeline) * frac))
    tail = timeline[-k:]
    emitted = sum(row.get("legit_emitted_step", 0) for row in tail)
    dur = max(1e-9, tail[-1]["time"] - tail[0]["time"] + 1.0)
    return emitted / dur


def compare_attack_vs_baseline(cfg: ServiceConfig) -> Dict:
    """Run no-attack baseline vs attack, return throughput-retention & TTFT blowup."""
    import copy
    base_cfg = copy.deepcopy(cfg); base_cfg.attack_enabled = False
    atk_cfg = copy.deepcopy(cfg); atk_cfg.attack_enabled = True
    base = LLMServiceSimulator(base_cfg).run()
    atk = LLMServiceSimulator(atk_cfg).run()
    b, a = base["summary"], atk["summary"]
    ret = (a["legit_throughput_tps"] / b["legit_throughput_tps"]) if b["legit_throughput_tps"] else float("nan")
    blow = (a["mean_legit_ttft"] / b["mean_legit_ttft"]) if b["mean_legit_ttft"] else float("nan")
    ss_base = steady_state_tps(base["timeline"])
    ss_atk = steady_state_tps(atk["timeline"])
    ss_ret = (ss_atk / ss_base) if ss_base else float("nan")
    return dict(baseline=b, attacked=a,
                throughput_retention=ret, ttft_blowup=blow,
                steady_state_retention=ss_ret,
                steady_state_tps_baseline=ss_base, steady_state_tps_attacked=ss_atk,
                baseline_timeline=base["timeline"], attacked_timeline=atk["timeline"])
