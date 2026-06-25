#!/usr/bin/env python3
"""Simulate the serving-system impact and evaluate defenses.

Reproduces (qualitatively) the paper's Sec. VII-D / VIII-B:
  * FIFO + attack  -> memory exhaustion, TTFT blowup, throughput collapse.
  * VTC defense    -> bounded memory, preserved legit throughput (single-user).
  * naive output caps (1024 vs 128) -> illustrate the QoS trade-off.

python scripts/run_service_sim.py --out results
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from thinktrap.service_sim import ServiceConfig, LLMServiceSimulator, compare_attack_vs_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--horizon", type=float, default=600.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scenarios = {
        # capable continuous-batching server (e.g., vLLM/MindIE-class)
        "fifo_no_defense": ServiceConfig(scheduler="fifo", horizon_s=args.horizon),
        "vtc_defense": ServiceConfig(scheduler="vtc", vtc_quantum=1024, horizon_s=args.horizon),
        "cap_1024": ServiceConfig(scheduler="fifo", output_cap=1024, horizon_s=args.horizon),
        "cap_128": ServiceConfig(scheduler="fifo", output_cap=128, horizon_s=args.horizon),
        # naive small-batch serving (paper Sec. VII-D: HF Transformers, few slots)
        # -- here a 1024-token cap barely helps, only an aggressive 128 cap stabilizes.
        "naive_cap_1024": ServiceConfig(scheduler="fifo", output_cap=1024,
                                        max_batch=2, capacity_tps=120.0, horizon_s=args.horizon),
        "naive_cap_128": ServiceConfig(scheduler="fifo", output_cap=128,
                                       max_batch=2, capacity_tps=120.0, horizon_s=args.horizon),
    }

    report = {}
    print("=" * 90)
    print(f"{'scenario':20s} {'avg_retain':>11s} {'steady_retain':>14s} "
          f"{'ttft_blowup':>12s} {'peak_mem_MB':>12s} {'crash':>6s}")
    print("-" * 90)
    for name, cfg in scenarios.items():
        cmp = compare_attack_vs_baseline(cfg)
        a = cmp["attacked"]
        report[name] = dict(
            throughput_retention=cmp["throughput_retention"],
            steady_state_retention=cmp["steady_state_retention"],
            ttft_blowup=cmp["ttft_blowup"],
            peak_gpu_mem_mb=a["peak_gpu_mem_mb"],
            any_crash=a["any_crash"],
            attacked_summary=a, baseline_summary=cmp["baseline"],
        )
        print(f"{name:20s} {cmp['throughput_retention']*100:10.1f}% "
              f"{cmp['steady_state_retention']*100:13.1f}% "
              f"{cmp['ttft_blowup']:11.1f}x {a['peak_gpu_mem_mb']:12.0f} "
              f"{str(a['any_crash']):>6s}")
    print("=" * 90)

    out = os.path.join(args.out, "service_sim_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {out}")

    # timeline plot for FIFO-no-defense vs VTC
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        for name in ["fifo_no_defense", "vtc_defense"]:
            cmp = compare_attack_vs_baseline(scenarios[name])
            tl = cmp["attacked_timeline"]
            t = [r["time"] for r in tl]
            ax[0].plot(t, [r["gpu_mem_mb"] for r in tl], label=name)
            ax[1].plot(t, [r["mean_legit_ttft"] for r in tl], label=name)
            ax[2].plot(t, [r["legit_throughput_tps"] for r in tl], label=name)
        ax[0].axhline(ServiceConfig().mem_cap_mb, ls="--", c="r", lw=1)
        ax[0].set_title("GPU/KV memory (MB)"); ax[0].set_xlabel("time (s)")
        ax[1].set_title("mean legit TTFT (s)"); ax[1].set_xlabel("time (s)")
        ax[2].set_title("legit throughput (tok/s)"); ax[2].set_xlabel("time (s)")
        for a in ax:
            a.legend(fontsize=8)
        plt.tight_layout()
        png = os.path.join(args.out, "service_impact.png")
        plt.savefig(png, dpi=130)
        print(f"saved -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
