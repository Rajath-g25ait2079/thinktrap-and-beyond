"""Configuration objects for the ThinkTrap research re-implementation.

This is a faithful, *defensive-research* re-implementation of:

    Li et al., "ThinkTrap: Denial-of-Service Attacks against Black-box LLM
    Services via Infinite Thinking", NDSS 2026.

It is intended to be run ONLY against local / open-source models that you
control in a sandbox, for the purpose of studying the vulnerability and
building defenses. See README.md (Ethics) before use.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class ThinkTrapConfig:
    """Hyper-parameters for the offline Attack-Prompt-Generation (APG) stage.

    Defaults mirror the values reported in the paper where given:
      * prompt_length L      = 20      (Sec. VII-B, swept over {10,20,30,40,50})
      * latent_dim    m      = 20      (Sec. VII-B, swept over {10,20,50,100,200})
      * max_new_tokens       = 4096    (paper's reported generation cap)
      * temperature          = 1.0     (default; robustness tested {0,0.7,1,1.7})
    CMA-ES population (N), top-k, sigma and epsilon are not given numerically in
    the paper; we expose them and default to standard CMA-ES choices.
    """

    # --- prompt / latent geometry (Eqs. 1-2) ---
    prompt_length: int = 20          # L : number of token positions in p^att
    latent_dim: int = 20             # m : dimension of the low-dim latent z
    embed_dim: Optional[int] = None  # d : token-embedding dim (set from surrogate)

    # --- objective / querying (Eqs. 4-5) ---
    max_new_tokens: int = 4096       # generation cap; success := output >= success_threshold
    success_threshold: int = 4096    # paper: a "successful attack" forces >= ~4k tokens
    temperature: float = 1.0

    # --- CMA-ES (Eqs. 6-8) ---
    population_size: Optional[int] = None  # N ; None -> 4 + floor(3 ln m)
    top_k: Optional[int] = None            # k ; None -> floor(N/2)
    sigma: float = 1.0                     # initial search radius (Sigma0 = sigma^2 I)
    epsilon: float = 1e-8                  # numerical-stability term in cov update

    # --- budget / stopping ---
    query_budget_tokens: int = 100_000     # paper sweeps 10k..100k
    max_iterations: int = 200

    # --- reproducibility ---
    seed: int = 0

    # --- victim / surrogate selection ---
    victim: Dict[str, Any] = field(default_factory=lambda: {"kind": "mock"})
    surrogate: Dict[str, Any] = field(default_factory=lambda: {"kind": "synthetic", "vocab_size": 2000})

    # --- logging ---
    log_every: int = 1
    verbose: bool = True

    def resolved_population(self, m: int) -> int:
        import math
        return self.population_size or (4 + int(3 * math.log(max(m, 2))))

    def resolved_topk(self, n: int) -> int:
        return self.top_k or max(1, n // 2)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_yaml(cls, path: str) -> "ThinkTrapConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
