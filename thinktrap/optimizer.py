"""Derivative-Free Optimization (DFO)  --  Sec. V-D, Eqs. (5)-(8).

The attack objective max_z L(z) = o^t (Eq. 5) is non-differentiable (it is the
output length returned by a black-box autoregressive model), so ThinkTrap uses
CMA-ES, which maintains a Gaussian search distribution N(mu^t, Sigma^t) over the
latent space:

    init:  mu^0 = 0,  Sigma^0 = sigma^2 I
    sample (Eq.6):   z_i ~ N(mu^t, Sigma^t),  i = 1..N
    evaluate:        o_i = M_vic(decode(A z_i))
    recombine (Eq.7): mu^{t+1} = sum_{j=1}^{k} w_j z_(j)            # top-k
    cov update (Eq.8):Sigma^{t+1} = sum w_j (z_(j)-mu')(z_(j)-mu')^T + eps I

Two backends:
  * cmaes library (CyberAgent/cmaes) if installed -- full CMA-ES.
  * SimpleCMAES (numpy) -- implements Eqs. (6)-(8) literally (rank-mu update),
    so the code runs with numpy alone and matches the paper's equations.

The optimizer is *maximization* over output length, and stops on any of:
query-token budget exhausted, success threshold reached, or max iterations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any
import numpy as np

# objective: z -> (score, token_cost, info)
Objective = Callable[[np.ndarray], Tuple[float, int, Dict[str, Any]]]


@dataclass
class OptimizeResult:
    best_z: np.ndarray
    best_score: float
    best_info: Dict[str, Any]
    history: List[dict] = field(default_factory=list)   # per-generation logs
    total_tokens: int = 0
    n_evaluations: int = 0
    converged_reason: str = ""


def _recombination_weights(k: int) -> np.ndarray:
    """Standard CMA-ES positive recombination weights, summing to 1."""
    j = np.arange(1, k + 1)
    w = np.log(k + 0.5) - np.log(j)
    w = np.clip(w, 0, None)
    return (w / w.sum()).astype(np.float64)


class SimpleCMAES:
    """Numpy implementation of the rank-mu CMA-ES described by Eqs. (6)-(8)."""

    def __init__(self, m: int, sigma: float, population_size: int, top_k: int,
                 epsilon: float, seed: int = 0):
        self.m = m
        self.N = population_size
        self.k = top_k
        self.eps = epsilon
        self.mu = np.zeros(m)                       # mu^0 = 0
        self.Sigma = (sigma ** 2) * np.eye(m)       # Sigma^0 = sigma^2 I
        self.w = _recombination_weights(top_k)
        self.rng = np.random.default_rng(seed)

    def ask(self) -> np.ndarray:
        # Eq.(6): sample N candidates from N(mu, Sigma)
        try:
            L = np.linalg.cholesky(self.Sigma + 1e-12 * np.eye(self.m))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(np.diag(np.diag(self.Sigma)) + 1e-9 * np.eye(self.m))
        Z = self.rng.normal(size=(self.N, self.m)) @ L.T + self.mu
        return Z

    def tell(self, Z: np.ndarray, scores: np.ndarray) -> None:
        order = np.argsort(-scores)                 # maximize -> descending
        top = Z[order[: self.k]]                    # z_(1..k)
        mu_new = (self.w[:, None] * top).sum(axis=0)        # Eq.(7)
        diff = top - mu_new
        Sigma_new = np.einsum("i,ij,ik->jk", self.w, diff, diff)  # Eq.(8) rank-mu
        Sigma_new += self.eps * np.eye(self.m)
        self.mu, self.Sigma = mu_new, Sigma_new


def _make_engine(m, cfg):
    N = cfg.resolved_population(m)
    k = cfg.resolved_topk(N)
    backend = "SimpleCMAES(numpy)"
    engine = None
    try:
        from cmaes import CMA  # type: ignore
        engine = CMA(mean=np.zeros(m), sigma=cfg.sigma, population_size=N, seed=cfg.seed)
        backend = "cmaes.CMA"
    except Exception:
        engine = SimpleCMAES(m, cfg.sigma, N, k, cfg.epsilon, seed=cfg.seed)
    return engine, backend, N, k


def optimize(objective: Objective, m: int, cfg) -> OptimizeResult:
    """Run CMA-ES to maximize the black-box objective `objective`."""
    engine, backend, N, k = _make_engine(m, cfg)
    best_z, best_score, best_info = None, -np.inf, {}
    history: List[dict] = []
    total_tokens = 0
    n_eval = 0
    reason = "max_iterations"

    using_lib = backend == "cmaes.CMA"
    for it in range(cfg.max_iterations):
        # ---- sample population (Eq. 6) ----
        if using_lib:
            Z = np.stack([engine.ask() for _ in range(N)])
        else:
            Z = engine.ask()

        # ---- evaluate ----
        scores = np.empty(len(Z))
        gen_best = -np.inf
        for i, z in enumerate(Z):
            score, cost, info = objective(z)
            scores[i] = score
            total_tokens += cost
            n_eval += 1
            if score > best_score:
                best_score, best_z, best_info = score, z.copy(), info
            gen_best = max(gen_best, score)
            if total_tokens >= cfg.query_budget_tokens:
                break

        # ---- update search distribution (Eqs. 7-8) ----
        evaluated = (~np.isnan(scores)) & (np.arange(len(Z)) < (i + 1))
        Ze, Se = Z[evaluated], scores[evaluated]
        if using_lib:
            try:
                engine.tell([(Ze[j], -float(Se[j])) for j in range(len(Ze))])
            except Exception:
                pass
        else:
            if len(Ze) >= k:
                engine.tell(Ze, Se)

        history.append(dict(iteration=it, gen_best=float(gen_best),
                            running_best=float(best_score),
                            cumulative_tokens=int(total_tokens),
                            n_evaluations=n_eval))
        if cfg.verbose and (it % cfg.log_every == 0):
            print(f"[APG] iter {it:3d} | gen_best={gen_best:7.1f} | "
                  f"best={best_score:7.1f} | tokens={total_tokens:>7d} | {backend}")

        if best_score >= cfg.success_threshold:
            reason = "success_threshold"
            break
        if total_tokens >= cfg.query_budget_tokens:
            reason = "budget_exhausted"
            break

    return OptimizeResult(best_z=best_z, best_score=best_score, best_info=best_info,
                          history=history, total_tokens=total_tokens,
                          n_evaluations=n_eval, converged_reason=reason)
