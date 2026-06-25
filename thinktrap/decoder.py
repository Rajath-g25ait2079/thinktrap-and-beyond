"""Surrogate Prompt Decoding (SPD)  --  Sec. V-B, Eq. (3).

The optimized embedding E in R^{L x d} is mapped back to a discrete token
sequence by nearest-neighbor search against a *surrogate* token-embedding
table T^sur in R^{|V| x d}:

    w_i = argmin_{j in V} || e_i - T^sur_j ||_2          (Eq. 3)

The paper exploits the empirical observation that token-embedding spaces across
models are highly aligned (shared tokenizers, overlapping corpora, convergent
training), so a *public* surrogate embedding table is a good stand-in for the
(inaccessible, black-box) victim's own embedding matrix.

Two backends:
  * SyntheticSurrogate : numpy-only random table for verification / CPU demos.
  * HFSurrogate        : real token-embedding matrix from a HuggingFace model
                         (requires `transformers` + `torch`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class DecodedPrompt:
    token_ids: np.ndarray   # (L,) int
    tokens: List[str]       # surrogate token strings
    text: str               # detokenized prompt text


class SurrogateDecoder:
    """Nearest-neighbor decoder over a fixed surrogate embedding table."""

    def __init__(self, embedding_table: np.ndarray, vocab: List[str], detokenize=None):
        self.T = np.asarray(embedding_table, dtype=np.float32)   # (|V|, d)
        self.vocab = vocab
        self.vocab_size, self.d = self.T.shape
        self._t_sqnorm = np.sum(self.T * self.T, axis=1)         # (|V|,)
        self._detok = detokenize or (lambda toks: " ".join(toks))

    def decode(self, E: np.ndarray) -> DecodedPrompt:
        """Eq. (3): map each of the L embeddings to its nearest token."""
        E = np.asarray(E, dtype=np.float32)               # (L, d)
        # squared euclidean distance: ||e||^2 - 2 e.T_j + ||T_j||^2
        cross = E @ self.T.T                              # (L, |V|)
        dists = (np.sum(E * E, axis=1, keepdims=True)
                 - 2.0 * cross
                 + self._t_sqnorm[None, :])               # (L, |V|)
        ids = np.argmin(dists, axis=1).astype(np.int64)   # (L,)
        toks = [self.vocab[i] for i in ids]
        return DecodedPrompt(token_ids=ids, tokens=toks, text=self._detok(toks))


def build_synthetic_surrogate(vocab_size: int = 2000, embed_dim: int = 64,
                              seed: int = 0) -> SurrogateDecoder:
    """A self-contained surrogate table for numpy-only experiments.

    Token strings are pseudo-words so decoded prompts are human-readable; the
    embedding geometry is what matters for exercising the z->E->tokens pipeline.
    """
    rng = np.random.default_rng(seed)
    table = rng.normal(0, 1.0 / np.sqrt(embed_dim), size=(vocab_size, embed_dim)).astype(np.float32)
    syllables = ["ka", "ti", "lo", "mu", "re", "no", "sa", "vi", "de", "po",
                 "tha", "gen", "rea", "son", "loop", "wait", "hmm", "step", "thus", "ergo"]
    vocab = []
    for i in range(vocab_size):
        w = "".join(rng.choice(syllables, size=rng.integers(1, 4)))
        vocab.append(f"{w}{i % 97}")
    return SurrogateDecoder(table, vocab)


def build_hf_surrogate(model_name: str, device: str = "cpu") -> SurrogateDecoder:
    """Real surrogate decoder from a HuggingFace model's input-embedding matrix.

    Requires `transformers` and `torch`. The returned decoder uses the model's
    tokenizer for detokenization so decoded prompts are valid for that family.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    emb = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    vocab_size = emb.shape[0]
    # vocab strings (for logging/inspection only)
    vocab = [tok.convert_ids_to_tokens(i) for i in range(vocab_size)]

    def detok(toks: List[str]) -> str:
        ids = tok.convert_tokens_to_ids(toks)
        return tok.decode(ids, skip_special_tokens=True)

    return SurrogateDecoder(emb, vocab, detokenize=detok)
