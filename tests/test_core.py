"""Unit + integration tests for the ThinkTrap re-implementation.

Run:  pytest -q      (or)   python tests/test_core.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thinktrap.projection import LowRankProjection
from thinktrap.decoder import build_synthetic_surrogate
from thinktrap.victim import MockVictim
from thinktrap.optimizer import optimize, SimpleCMAES
from thinktrap.config import ThinkTrapConfig
from thinktrap.attack import ThinkTrapAttack
from thinktrap import baselines
from thinktrap.defenses import NGramRepetitionDetector
from thinktrap.service_sim import ServiceConfig, compare_attack_vs_baseline


def test_projection_shapes_and_variance():
    L, d, m = 10, 64, 20
    proj = LowRankProjection(L, d, m, seed=0)
    assert proj.A.shape == (L * d, m)
    # entries ~ N(0, 1/m): empirical variance close to 1/m
    assert abs(proj.A.var() - 1.0 / m) < 0.02
    E = proj.project(np.ones(m))
    assert E.shape == (L, d)
    # batch projection consistent with single projection
    Z = np.random.default_rng(1).normal(size=(5, m))
    EB = proj.project_batch(Z)
    assert EB.shape == (5, L, d)
    assert np.allclose(EB[0], proj.project(Z[0]), atol=1e-4)
    print("OK projection: shapes, variance, isotropy_err=%.4f" % proj.isotropy_error())


def test_decoder_nearest_neighbor_exact():
    dec = build_synthetic_surrogate(vocab_size=500, embed_dim=32, seed=0)
    # build E from exact table rows -> decode must return those rows
    chosen = np.array([3, 17, 200, 499, 42])
    E = dec.T[chosen]
    out = dec.decode(E)
    assert np.array_equal(out.token_ids, chosen), (out.token_ids, chosen)
    assert len(out.tokens) == len(chosen)
    print("OK decoder: exact nearest-neighbor recovery, text=%r" % out.text[:40])


def test_cmaes_maximizes_quadratic():
    # maximize -||z - target||^2  (optimum at z=target, score 0)
    rng = np.random.default_rng(0)
    target = rng.normal(size=8)

    def obj(z):
        score = -float(np.sum((z - target) ** 2))
        return score, 1, {}

    cfg = ThinkTrapConfig(latent_dim=8, max_iterations=80, query_budget_tokens=10**9,
                          success_threshold=-1e-3, verbose=False, sigma=2.0)
    res = optimize(obj, 8, cfg)
    assert res.best_score > -0.5, res.best_score  # got close to optimum
    print("OK CMA-ES: best_score=%.4f after %d evals" % (res.best_score, res.n_evaluations))


def test_simple_cmaes_fallback_runs():
    es = SimpleCMAES(m=5, sigma=1.0, population_size=10, top_k=5, epsilon=1e-8, seed=0)
    Z = es.ask()
    assert Z.shape == (10, 5)
    es.tell(Z, np.random.default_rng(0).normal(size=10))
    assert es.Sigma.shape == (5, 5)
    print("OK SimpleCMAES numpy fallback")


def test_end_to_end_thinktrap_beats_random_init():
    cfg = ThinkTrapConfig(prompt_length=20, latent_dim=20, max_new_tokens=512,
                          success_threshold=10_000, query_budget_tokens=40_000,
                          max_iterations=80, verbose=False, seed=0)
    dec = build_synthetic_surrogate(vocab_size=2000, embed_dim=64, seed=0)
    vic = MockVictim(vocab_size=2000, max_new_tokens=512, seed=0)

    # reference: output length at random latent vectors (z ~ N(0, I))
    rng = np.random.default_rng(0)
    proj = LowRankProjection(20, 64, 20, seed=0)
    rand_lengths = []
    for _ in range(40):
        ids = dec.decode(proj.project(rng.normal(size=20))).token_ids
        rand_lengths.append(vic.query(ids))
    rand_mean = float(np.mean(rand_lengths))

    attack = ThinkTrapAttack(vic, dec, cfg)
    art = attack.run()
    assert art.output_length > rand_mean * 1.3, (art.output_length, rand_mean)
    print("OK end-to-end: ThinkTrap=%d vs random-init mean=%.1f (cap=512)"
          % (art.output_length, rand_mean))


def test_baselines_run_within_budget():
    dec = build_synthetic_surrogate(vocab_size=1000, embed_dim=48, seed=1)
    vic = MockVictim(vocab_size=1000, max_new_tokens=512, seed=1)
    ctx = baselines.BaselineContext(
        query_length_ids=lambda ids: vic.query(ids),
        vocab_size=1000, prompt_length=20, budget_tokens=15_000, seed=1)
    res = baselines.run_all_baselines(ctx)
    for k, r in res.items():
        assert r.best_length >= 1
        assert r.tokens_spent <= ctx.budget_tokens + 1000  # allow last-eval overshoot
    print("OK baselines:", {k: r.best_length for k, r in res.items()})


def test_ngram_detector_catches_loops_but_not_redundant():
    det = NGramRepetitionDetector(n=4, window=128, max_repeats=4)
    vic = MockVictim(vocab_size=500, max_new_tokens=300, seed=2)
    ids = np.arange(20)
    loop = vic.generate(ids, style="loop").output_token_ids
    red = vic.generate(ids, style="redundant").output_token_ids
    r_loop = det.scan(loop)
    r_red = det.scan(red)
    assert r_loop.terminated, "n-gram detector should catch exact loops"
    # redundant (ThinkTrap-like) is harder; it should survive much longer
    assert r_red.effective_length >= r_loop.effective_length
    print("OK defense: loop terminated@%d, redundant survives to %d"
          % (r_loop.effective_length, r_red.effective_length))


def test_service_sim_shows_degradation_and_vtc_helps():
    fifo = ServiceConfig(scheduler="fifo", horizon_s=300.0)
    vtc = ServiceConfig(scheduler="vtc", vtc_quantum=1024, horizon_s=300.0)
    c_fifo = compare_attack_vs_baseline(fifo)
    c_vtc = compare_attack_vs_baseline(vtc)
    # FIFO under attack should retain much less throughput than VTC
    assert c_fifo["throughput_retention"] <= c_vtc["throughput_retention"] + 1e-6
    print("OK service-sim: FIFO retain=%.1f%%  VTC retain=%.1f%%"
          % (c_fifo["throughput_retention"] * 100, c_vtc["throughput_retention"] * 100))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print("\n%d/%d tests passed" % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
