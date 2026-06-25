#!/usr/bin/env python3
"""Minimal end-to-end example: generate one ThinkTrap prompt on a mock victim.

    python examples/quickstart.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thinktrap import (ThinkTrapConfig, ThinkTrapAttack, MockVictim,
                       build_synthetic_surrogate)

# 1) a synthetic black-box victim (no model download) and a surrogate decoder
decoder = build_synthetic_surrogate(vocab_size=2000, embed_dim=64, seed=0)
victim = MockVictim(vocab_size=2000, max_new_tokens=512, seed=0)

# 2) configure the offline Attack-Prompt-Generation
cfg = ThinkTrapConfig(prompt_length=20, latent_dim=20, max_new_tokens=512,
                      success_threshold=500, query_budget_tokens=40_000,
                      max_iterations=80, verbose=True, seed=0)

# 3) run it
attack = ThinkTrapAttack(victim, decoder, cfg)
artifact = attack.run()

print("\n--- result ---")
print("best output length :", artifact.output_length, "/", cfg.max_new_tokens)
print("tokens spent       :", artifact.tokens_spent)
print("stop reason        :", artifact.converged_reason)
print("adversarial prompt :", artifact.prompt_text[:200])
