ThinkTrap and Beyond
Reproducing reasoning-induced denial of service on black-box LLMs, and
studying defenses.

This repo holds the code, paper, and slides for our project (Group 11,
IIT Jodhpur). It reproduces the attack from:

  Yunzhe Li, Jianan Wang, Hongzi Zhu, James Lin, Shan Chang, Minyi Guo.
  "ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services
  via Infinite Thinking." NDSS Symposium 2026.

  Paper:    https://www.ndss-symposium.org/wp-content/uploads/2026-f639-paper.pdf
  Slides:   https://www.ndss-symposium.org/wp-content/uploads/f0639-li-slides.pdf
  Abstract: https://www.ndss-symposium.org/ndss-paper/thinktrap-denial-of-service-attacks-against-black-box-llm-services-via-infinite-thinking/


WHAT THE ATTACK DOES

ThinkTrap finds a short adversarial prompt that makes a black-box LLM keep
"thinking" for a huge, almost unbounded number of tokens. That drains the
backend GPU and KV cache, which slows down or blocks service for everyone
else. The search is fully black-box: it optimizes a small latent vector,
projects it into the prompt-embedding space, decodes it to real tokens with
a surrogate vocabulary, and scores it using only the length of the model's
reply. It needs no gradients, no logits, and no access to model internals.

We re-implemented the method so the vulnerability, and the defenses against
it, can be measured in a repeatable way.


ETHICS AND RESPONSIBLE USE (PLEASE READ)

This is offensive-security code written for defensive work. Use it only on
models you own or are allowed to test, in an isolated setup. The configs
target local, open-source models. Do not point it at third-party or
commercial LLM APIs, and do not run it against any shared or production
service. The default target is a synthetic stand-in that needs no model at
all. If you ever find a real deployment that is affected, report it to the
provider's security team and give them time to fix it before saying anything
publicly. Keep any generated attack prompts out of public repositories. You
are responsible for following the law, the relevant terms of service, and
your institution's ethics rules.


REPOSITORY CONTENTS

  thinktrap/                  the re-implementation (library, configs,
                              scripts, tests, examples)
  paper/                      our follow-up write-up (LaTeX source and the
                              compiled PDF) with its figures
  research/                   research plan, paper outline, and a summary of
                              the original work
  ThinkTrap_and_Beyond.pptx   the presentation slides


INSTALL

  cd thinktrap
  pip install -r requirements.txt

The core (optimizer, projection, decoder, baselines, defenses, simulator,
and tests) runs on numpy alone. torch and transformers are only needed if
you want to run against a real Hugging Face model.


QUICKSTART (no model download, runs in seconds)

  cd thinktrap

  # attack-prompt search against a synthetic black box, vs the baselines
  python scripts/run_attack.py --config configs/cpu_demo.yaml

  # serving-system impact and defense comparison (FIFO vs VTC vs caps)
  python scripts/run_service_sim.py --out results

  # tests
  python tests/test_core.py

run_attack.py writes results/attack_results.json and results/convergence.png
(output length against query budget, for our search and each baseline).
run_service_sim.py writes results/service_sim_results.json and
results/service_impact.png (memory, time to first token, and throughput over
time).


RUNNING AGAINST A REAL LOCAL MODEL

Edit configs/gpu.yaml so the victim and surrogate point at a local Hugging
Face causal LM that you control (a small Qwen, Llama, or DeepSeek distill
works), then:

  python scripts/run_attack.py --config configs/gpu.yaml

The victim is treated as a pure black box; only the reply length is read. The
surrogate decoder can be the same model or a different public one, which lets
you test the cross-model assumption from the paper.


HOW THE CODE LINES UP WITH THE PAPER

  projection.py    low-rank embedding projection (Sec. V-A, Eqs. 1-2)
  decoder.py       surrogate nearest-neighbor decoding (Sec. V-B, Eq. 3)
  victim.py        black-box query returning only output length (V-C, Eq. 4)
  attack.py        objective and the offline search loop (Eq. 5; Sec. IV, VI)
  optimizer.py     CMA-ES, with a numpy fallback (Sec. V-D, Eqs. 6-8)
  baselines.py     decoy, semantic, sponge, effichecker, random (VII-A-3)
  metrics.py       output length, throughput, TTFT, GPU memory (VII-A-4)
  defenses.py      n-gram repetition detector and VTC scheduling (Sec. VIII)
  service_sim.py   serving-system impact simulator (Sec. VII-D, VIII-B)

The offline search starts CMA-ES at mu0 = 0 and Sigma0 = sigma^2 I. Each
round it samples N latent vectors, projects each to an embedding with a fixed
Gaussian matrix, decodes to a prompt by nearest neighbor in the surrogate
table, queries the target for the reply length, and then updates the search
mean and covariance from the best candidates. It stops when a long-enough
output is found, the query budget runs out, or it hits the iteration limit.


WHAT THE DEMO SHOWS

The search drives output length to the cap and beats the decoy and semantic
prompts, random search, a sponge-style genetic search, and an effichecker-
style heuristic at the same query budget. In the serving simulator, a
low-rate attack (about 10 requests per minute) drops legitimate throughput to
roughly 20 percent in the default setting, pushes time to first token up by
about 200 times, and fills KV memory until the server crashes. For defenses,
a 4-gram repetition detector stops naive looping output but misses the
semantically varied output this attack produces; VTC fair scheduling keeps
throughput up against a single attacker. A plain output cap depends on the
server: a 1024-token cap protects a capable continuous-batching server but
does little on a naive small-batch server, where only a tight 128-token cap
holds, and that hurts normal long answers.


NOTES AND DIFFERENCES FROM THE PAPER

The default target is synthetic (a sparse "verbosity" objective with a small
bigram-loop bonus) so the whole pipeline runs with no model and no network;
switch to HFVictim for real measurements. The serving simulator is an
abstraction (continuous batching, a fixed generation rate, and memory that
grows with the KV cache); it reproduces the paper's qualitative effects, and
the absolute numbers depend on the settings you pick. The paper does not give
exact CMA-ES settings, so we use standard ones, with the cmaes library when
it is installed and a numpy version of the update equations otherwise. The
paper does not name the surrogate model either; any aligned public embedding
table works.


CITATION

If you build on this, please cite the original ThinkTrap paper (NDSS 2026).
This repository is an independent re-implementation for defensive research.
