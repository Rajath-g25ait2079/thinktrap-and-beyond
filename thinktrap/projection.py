"""Low-rank Embedding Projection (LEP)  --  Sec. V-A, Eqs. (1)-(2).

ThinkTrap optimizes in a low-dimensional latent space R^m instead of the full
prompt-embedding space R^{L*d}, exploiting the redundancy / low intrinsic
dimensionality of LLM input spaces.

A fixed, random Gaussian projection matrix A in R^{(L*d) x m} maps a latent
vector z in R^m to a full embedding E = A z in R^{L*d}, which is then reshaped
to L token-embeddings of dimension d.

    A_{i,j} ~ N(0, 1/m)            (Eq. 1)
    E       = A z                  (Eq. 2)

The N(0,1/m) scaling makes the projection approximately isotropic and
norm-preserving in expectation (a Johnson-Lindenstrauss-style embedding),
satisfying the paper's three design criteria: isotropy, no coordinate
amplification, and diversity (distinct z -> distinct E).
"""
from __future__ import annotations

import numpy as np


class LowRankProjection:
    def __init__(self, prompt_length: int, embed_dim: int, latent_dim: int, seed: int = 0):
        self.L = int(prompt_length)
        self.d = int(embed_dim)
        self.m = int(latent_dim)
        self.full_dim = self.L * self.d
        rng = np.random.default_rng(seed)
        # Eq. (1): entries i.i.d. Gaussian, variance 1/m.
        self.A = rng.normal(0.0, np.sqrt(1.0 / self.m), size=(self.full_dim, self.m)).astype(np.float32)

    def project(self, z: np.ndarray) -> np.ndarray:
        """Eq. (2): z in R^m  ->  E in R^{L x d}."""
        z = np.asarray(z, dtype=np.float32).reshape(self.m)
        e_flat = self.A @ z                      # R^{L*d}
        return e_flat.reshape(self.L, self.d)    # reshape to L tokens x d

    def project_batch(self, Z: np.ndarray) -> np.ndarray:
        """Project a batch Z in R^{N x m} -> R^{N x L x d}."""
        Z = np.asarray(Z, dtype=np.float32)
        E = Z @ self.A.T                         # R^{N x (L*d)}
        return E.reshape(Z.shape[0], self.L, self.d)

    # --- diagnostics used by the unit tests ---
    def isotropy_error(self, n_probe: int = 256, seed: int = 1) -> float:
        """Mean abs deviation of ||Az|| / ||z|| from 1 over random unit z.

        For A_{ij} ~ N(0, 1/m), E[||Az||^2] = (Ld/m)||z||^2, so we compare the
        *direction-preserving* normalized gram to identity instead.
        """
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(n_probe, self.m)).astype(np.float32)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        E = (Z @ self.A.T)
        norms = np.linalg.norm(E, axis=1) / np.sqrt(self.full_dim / self.m)
        return float(np.mean(np.abs(norms - 1.0)))
