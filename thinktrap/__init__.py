"""ThinkTrap -- defensive-research re-implementation (NDSS 2026).

Faithful, modular re-implementation of the ThinkTrap reasoning-induced DoS
attack on black-box LLMs, for studying the vulnerability and building defenses.
Run ONLY against local / open-source models you control. See README (Ethics).
"""
from .config import ThinkTrapConfig
from .projection import LowRankProjection
from .decoder import (SurrogateDecoder, build_synthetic_surrogate,
                      build_hf_surrogate, DecodedPrompt)
from .victim import (BaseVictim, MockVictim, HFVictim, build_victim, GenerationResult)
from .optimizer import optimize, OptimizeResult, SimpleCMAES
from .attack import ThinkTrapAttack, AttackArtifact, save_artifacts
from . import baselines, defenses, metrics, service_sim

__version__ = "0.1.0"
__all__ = [
    "ThinkTrapConfig", "LowRankProjection", "SurrogateDecoder",
    "build_synthetic_surrogate", "build_hf_surrogate", "DecodedPrompt",
    "BaseVictim", "MockVictim", "HFVictim", "build_victim", "GenerationResult",
    "optimize", "OptimizeResult", "SimpleCMAES",
    "ThinkTrapAttack", "AttackArtifact", "save_artifacts",
    "baselines", "defenses", "metrics", "service_sim",
]
