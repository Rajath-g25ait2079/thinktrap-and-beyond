"""LLM Querying (LQ)  --  Sec. V-C, Eq. (4).

The victim model is treated as a pure black box: the attacker submits a discrete
prompt and observes ONLY the number of generated tokens (the optimization
objective o^t):

    o^t = M_vic(p^t)                         (Eq. 4)

Backends
--------
* MockVictim : numpy-only, deterministic+noisy synthetic length objective with
               *exploitable structure* (a sparse set of "verbose trigger"
               tokens, plus a bigram-loop bonus), so the full
               z -> E -> tokens -> length pipeline can be verified and CMA-ES
               can be shown to climb -- WITHOUT downloading any model.
               It also emits synthetic *output token streams* (repetitive or
               semantically-redundant) used by the defense experiments.
* HFVictim   : a real local HuggingFace causal-LM. Returns output length and,
               for the service/defense harness, TTFT / total-time / text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import numpy as np


@dataclass
class GenerationResult:
    output_length: int
    output_token_ids: Optional[np.ndarray] = None
    text: Optional[str] = None
    ttft_s: Optional[float] = None       # time to first token
    total_time_s: Optional[float] = None
    terminated_by_defense: bool = False


class BaseVictim:
    def query(self, token_ids: np.ndarray, text: Optional[str] = None) -> int:
        """Return ONLY the output length (Eq. 4) -- the single feedback signal."""
        raise NotImplementedError

    def generate(self, token_ids: np.ndarray, text: Optional[str] = None) -> GenerationResult:
        """Richer call used by the service/defense harness (not the attacker)."""
        raise NotImplementedError


class MockVictim(BaseVictim):
    """Synthetic black-box LLM with a learnable 'verbosity' structure.

    Design goals:
      * length is a function of the *decoded tokens*, so the whole pipeline is
        exercised by the optimizer;
      * structure is *sparse* (few high-verbosity trigger tokens) mirroring the
        paper's "input sparsity" assumption;
      * a bigram-loop bonus rewards prompts that put the model into a
        self-reinforcing loop -> approaches the max_new_tokens cap.
    """

    def __init__(self, vocab_size: int = 2000, max_new_tokens: int = 4096,
                 base_len: int = 40, scale: float = 6.0, noise_std: float = 8.0,
                 frac_trigger: float = 0.04, seed: int = 0):
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self.base_len = base_len
        self.scale = scale
        self.noise_std = noise_std
        rng = np.random.default_rng(seed)
        # Sparse verbosity: most tokens ~0, a few "trigger" tokens near 1.
        v = rng.beta(0.25, 4.0, size=vocab_size).astype(np.float32)
        n_trig = max(1, int(frac_trigger * vocab_size))
        trig = rng.choice(vocab_size, size=n_trig, replace=False)
        v[trig] = rng.uniform(0.8, 1.0, size=n_trig).astype(np.float32)
        self.verbosity = v
        # bigram-loop affinity: random matrix; high values => looping bonus.
        self._bg = rng.random((vocab_size, vocab_size), dtype=np.float32) if vocab_size <= 4096 \
            else None
        self._bg_seed = seed
        self._rng = rng

    def _bigram_bonus(self, ids: np.ndarray) -> float:
        if len(ids) < 2:
            return 0.0
        if self._bg is not None:
            pairs = self._bg[ids[:-1], ids[1:]]
            return float(np.sum(pairs > 0.95)) * 60.0
        # memory-light fallback: hash-based pseudo-affinity
        h = ((ids[:-1] * 1103515245 + ids[1:] * 12345) % 1000) / 1000.0
        return float(np.sum(h > 0.95)) * 60.0

    def _expected_length(self, ids: np.ndarray) -> float:
        verb = float(np.sum(self.verbosity[ids]))
        length = self.base_len + self.scale * verb * (self.max_new_tokens / 60.0) / max(len(ids), 1)
        length += self._bigram_bonus(ids)
        return length

    def query(self, token_ids: np.ndarray, text: Optional[str] = None) -> int:
        ids = np.asarray(token_ids, dtype=np.int64)
        length = self._expected_length(ids)
        length += self._rng.normal(0, self.noise_std)
        return int(np.clip(round(length), 1, self.max_new_tokens))

    def generate(self, token_ids: np.ndarray, text: Optional[str] = None,
                 style: str = "redundant") -> GenerationResult:
        """Emit a synthetic output stream for defense experiments.

        style="loop"      : exact 4-gram repetition (naive sponge; n-gram
                            detectors catch this).
        style="redundant" : semantically-redundant but non-repeating stream
                            (ThinkTrap-like; evades n-gram detectors).
        """
        ids = np.asarray(token_ids, dtype=np.int64)
        n = self.query(ids)
        rng = np.random.default_rng(int(abs(hash((ids.tobytes(), style))) % (2**32)))
        if style == "loop":
            motif = rng.integers(0, self.vocab_size, size=4)
            out = np.tile(motif, n // 4 + 1)[:n]
        else:  # redundant: drift slowly through a pool of "reasoning" tokens
            pool = rng.integers(0, self.vocab_size, size=64)
            out = rng.choice(pool, size=n)
        # synthetic timing (proportional to length); ~50 tok/s nominal
        total = n / 50.0
        return GenerationResult(output_length=n, output_token_ids=out,
                                ttft_s=0.04, total_time_s=total)


class HFVictim(BaseVictim):
    """Real local HuggingFace causal-LM victim (requires transformers+torch)."""

    def __init__(self, model_name: str, max_new_tokens: int = 4096,
                 temperature: float = 1.0, device: str = "auto", dtype: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _encode(self, token_ids, text):
        import numpy as _np
        if text is not None:
            enc = self.tok(text, return_tensors="pt")
            return enc["input_ids"].to(self.device)
        ids = _np.asarray(token_ids, dtype="int64")
        return self.torch.tensor(ids[None, :], device=self.device)

    def query(self, token_ids: np.ndarray, text: Optional[str] = None) -> int:
        return self.generate(token_ids, text).output_length

    def generate(self, token_ids: np.ndarray, text: Optional[str] = None) -> GenerationResult:
        torch = self.torch
        input_ids = self._encode(token_ids, text)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens,
                          do_sample=self.temperature > 0,
                          temperature=max(self.temperature, 1e-5),
                          pad_token_id=self.tok.eos_token_id)
        t0 = time.perf_counter()
        ttft = None
        try:
            from transformers import TextIteratorStreamer  # type: ignore
            from threading import Thread
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True)
            th = Thread(target=self.model.generate,
                        kwargs=dict(inputs=input_ids, streamer=streamer, **gen_kwargs))
            th.start()
            pieces, first = [], True
            for piece in streamer:
                if first:
                    ttft = time.perf_counter() - t0
                    first = False
                pieces.append(piece)
            th.join()
            text_out = "".join(pieces)
            out_ids = self.tok(text_out, return_tensors="np")["input_ids"][0]
            n = int(len(out_ids))
        except Exception:
            with torch.no_grad():
                out = self.model.generate(input_ids, **gen_kwargs)
            out_ids = out[0, input_ids.shape[1]:].detach().cpu().numpy()
            n = int(len(out_ids))
            text_out = self.tok.decode(out_ids, skip_special_tokens=True)
        total = time.perf_counter() - t0
        return GenerationResult(output_length=n, output_token_ids=out_ids,
                                text=text_out, ttft_s=ttft, total_time_s=total)


def build_victim(spec: dict, max_new_tokens: int = 4096, temperature: float = 1.0,
                 vocab_size: int = 2000, seed: int = 0) -> BaseVictim:
    kind = spec.get("kind", "mock")
    if kind == "mock":
        return MockVictim(vocab_size=spec.get("vocab_size", vocab_size),
                          max_new_tokens=max_new_tokens, seed=seed)
    if kind == "hf":
        return HFVictim(spec["model_name"], max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        device=spec.get("device", "auto"),
                        dtype=spec.get("dtype", "auto"))
    raise ValueError(f"unknown victim kind: {kind}")
