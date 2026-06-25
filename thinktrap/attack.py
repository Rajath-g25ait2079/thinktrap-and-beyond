"""ThinkTrap orchestration  --  Sec. IV-VI.

Stage 1 (offline) -- Attack Prompt Generation (APG):
    loop:  z --LEP--> E --SPD--> p --LQ--> o   guided by  DFO (CMA-ES)
    i.e.   the four submodules of Sec. V run in a loop until the query budget
    is exhausted or a successful (long-output) prompt is found.

Stage 2 (online) -- Denial-of-Service Attack (DSA):
    the pre-generated prompt pool is replayed against the live service at a low
    rate (see thinktrap.service_sim for the serving-system impact harness).

This module exposes `ThinkTrapAttack`, the offline generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import os
import numpy as np

from .config import ThinkTrapConfig
from .projection import LowRankProjection
from .decoder import SurrogateDecoder, DecodedPrompt
from .victim import BaseVictim
from . import optimizer as dfo


@dataclass
class AttackArtifact:
    prompt_text: str
    token_ids: List[int]
    output_length: int
    latent_z: List[float]
    tokens_spent: int
    converged_reason: str
    history: List[dict] = field(default_factory=list)
    backend: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


class ThinkTrapAttack:
    """Offline Attack-Prompt-Generation (APG) for a single black-box victim."""

    def __init__(self, victim: BaseVictim, decoder: SurrogateDecoder,
                 cfg: ThinkTrapConfig):
        self.victim = victim
        self.decoder = decoder
        self.cfg = cfg
        d = decoder.d
        cfg.embed_dim = d
        # LEP: fixed random Gaussian projection (Eqs. 1-2)
        self.projection = LowRankProjection(cfg.prompt_length, d, cfg.latent_dim, seed=cfg.seed)
        self._eval_count = 0

    # --- the objective L(z) = o^t (Eqs. 2-5) ---
    def _objective(self, z: np.ndarray):
        E = self.projection.project(z)              # Eq.2  : z -> E (L x d)
        decoded: DecodedPrompt = self.decoder.decode(E)  # Eq.3 : E -> tokens p
        o = self.victim.query(decoded.token_ids, decoded.text)  # Eq.4 : o = M_vic(p)
        # attacker pays prompt tokens + generated tokens per query
        cost = int(self.cfg.prompt_length + o)
        self._eval_count += 1
        info = {"text": decoded.text, "token_ids": decoded.token_ids.tolist(),
                "output_length": int(o)}
        return float(o), cost, info             # Eq.5 : maximize o

    def run(self) -> AttackArtifact:
        res = dfo.optimize(self._objective, self.cfg.latent_dim, self.cfg)
        backend = res.history[0] if res.history else {}
        artifact = AttackArtifact(
            prompt_text=res.best_info.get("text", ""),
            token_ids=res.best_info.get("token_ids", []),
            output_length=int(res.best_score),
            latent_z=np.asarray(res.best_z).tolist() if res.best_z is not None else [],
            tokens_spent=res.total_tokens,
            converged_reason=res.converged_reason,
            history=res.history,
        )
        return artifact

    def run_pool(self, n_prompts: int = 5) -> List[AttackArtifact]:
        """Generate a pool of attack prompts (different CMA-ES seeds)."""
        pool = []
        base_seed = self.cfg.seed
        for i in range(n_prompts):
            self.cfg.seed = base_seed + i
            self.projection = LowRankProjection(self.cfg.prompt_length, self.decoder.d,
                                                self.cfg.latent_dim, seed=self.cfg.seed)
            pool.append(self.run())
        self.cfg.seed = base_seed
        return pool


def save_artifacts(artifacts: List[AttackArtifact], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([a.to_dict() for a in artifacts], f, indent=2)
