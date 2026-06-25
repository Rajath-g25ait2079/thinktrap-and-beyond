#!/usr/bin/env python3
"""Run the offline Attack-Prompt-Generation (APG) and compare with baselines.

Examples
--------
# numpy-only demo (no model download), small budget:
python scripts/run_attack.py --config configs/cpu_demo.yaml

# against a real local HuggingFace model (needs transformers+torch+GPU):
python scripts/run_attack.py --config configs/gpu.yaml

Outputs JSON (history + best prompts) and, if matplotlib is available, a
convergence plot, into results/.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from thinktrap import (ThinkTrapConfig, ThinkTrapAttack, build_victim,
                       build_synthetic_surrogate, build_hf_surrogate, baselines)


def get_decoder(cfg):
    spec = cfg.surrogate
    if spec.get("kind") == "hf":
        return build_hf_surrogate(spec["model_name"])
    return build_synthetic_surrogate(vocab_size=spec.get("vocab_size", 2000),
                                     embed_dim=spec.get("embed_dim", 64),
                                     seed=cfg.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--budget", type=int, default=None, help="override query_budget_tokens")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    cfg = ThinkTrapConfig.from_yaml(args.config) if args.config else ThinkTrapConfig()
    if args.budget:
        cfg.query_budget_tokens = args.budget
    os.makedirs(args.out, exist_ok=True)

    decoder = get_decoder(cfg)
    cfg.embed_dim = decoder.d
    victim = build_victim(cfg.victim, max_new_tokens=cfg.max_new_tokens,
                          temperature=cfg.temperature, vocab_size=decoder.vocab_size,
                          seed=cfg.seed)

    print("=" * 70)
    print(f"ThinkTrap APG | L={cfg.prompt_length} m={cfg.latent_dim} d={decoder.d} "
          f"|V|={decoder.vocab_size} budget={cfg.query_budget_tokens} cap={cfg.max_new_tokens}")
    print("=" * 70)

    attack = ThinkTrapAttack(victim, decoder, cfg)
    artifact = attack.run()
    print(f"\n[ThinkTrap] best output length = {artifact.output_length} "
          f"(/{cfg.max_new_tokens}); tokens spent = {artifact.tokens_spent}; "
          f"reason = {artifact.converged_reason}")
    print(f"[ThinkTrap] best prompt: {artifact.prompt_text[:160]!r}")

    results = {"config": json.loads(cfg.to_json()),
               "thinktrap": artifact.to_dict()}

    if not args.skip_baselines:
        ctx = baselines.BaselineContext(
            query_length_ids=lambda ids: victim.query(ids),
            vocab_size=decoder.vocab_size,
            prompt_length=cfg.prompt_length,
            budget_tokens=cfg.query_budget_tokens,
            seed=cfg.seed,
            query_length_text=(lambda t: victim.query(np.array([]), t)) if cfg.victim.get("kind") == "hf" else None,
        )
        bl = baselines.run_all_baselines(ctx)
        print("\n--- Baselines (best output length / budget) ---")
        for k, r in bl.items():
            print(f"  {r.name:24s}: {r.best_length:5d}  (tokens={r.tokens_spent})")
        results["baselines"] = {k: r.__dict__ for k, r in bl.items()}

    out_json = os.path.join(args.out, "attack_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_json}")

    # convergence plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hist = artifact.history
        x = [h["cumulative_tokens"] for h in hist]
        y = [h["running_best"] for h in hist]
        plt.figure(figsize=(7, 4.2))
        plt.plot(x, y, "-o", ms=3, label="ThinkTrap (CMA-ES)")
        if not args.skip_baselines:
            for k, r in bl.items():
                hx = [h.get("cumulative_tokens") for h in r.history]
                hy = [h.get("best") for h in r.history]
                if hx and hy:
                    plt.plot(hx, hy, alpha=0.7, label=r.name)
        plt.axhline(cfg.success_threshold, ls="--", c="gray", lw=1, label="success threshold")
        plt.xlabel("query budget (tokens)"); plt.ylabel("best output length")
        plt.title("Output length vs. query budget"); plt.legend(fontsize=8)
        plt.tight_layout()
        png = os.path.join(args.out, "convergence.png")
        plt.savefig(png, dpi=130)
        print(f"saved -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
