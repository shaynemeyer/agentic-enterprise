# Choosing an Embedding Model — qwen3-embedding vs. nomic-embed-text-v2-moe

**Update (2026-09-03): this call is being revisited.** `nomic-embed-text-v2-moe`
is a custom MoE architecture that needs `trust_remote_code=True` to load. That
already makes it a narrower fit for portable, backend-agnostic serving than a
plain dense architecture — concretely, a request to add support to vLLM was
closed as not planned
([vllm-project/vllm#15849](https://github.com/vllm-project/vllm/issues/15849)),
and custom/remote-code architectures are exactly the class of model that
inference servers other than the one it was built for tend to lag on or skip
entirely. See [§5](#5-architecture-portability-a-filter-before-mteb-rank)
below — portability across OpenAI-compatible servers is now checked before
anything else, not after picking on quality/cost. The current front-runner is
`qwen3-embedding:0.6b`, a plain dense architecture. §1 keeps the original
two-model comparison for reference, with 0.6B added as the entry that's
actually portable.

Original framing: we picked `nomic-embed-text-v2-moe` for the semantic
memory store over `qwen3-embedding:8b`. This doc is the comparison behind
that call, plus the general rules worth applying the next time an embedding
model has to be chosen — for this repo's memory store or anything else.

---

## 1. The Models, Side by Side

|                                       | `qwen3-embedding:0.6b`                     | `qwen3-embedding:8b`           | `nomic-embed-text-v2-moe`                                 |
| ------------------------------------- | ------------------------------------------- | ------------------------------- | ----------------------------------------------------------- |
| Ollama pull tag                       | `qwen3-embedding:0.6b` (official `library/`) | `qwen3-embedding:8b`           | `nomic-embed-text-v2-moe:latest`                            |
| Model size on disk                    | 639MB (Q8_0) / 1.2GB (F16)                 | 4.7GB                          | 958MB                                                       |
| Parameters                            | 0.6B (dense)                               | 7.57B (dense)                  | 475M total / 305M active (MoE, 8 experts, top-2 routing)    |
| Output dimensions                     | 1024 default, MRL down to 32               | 4096 default, MRL down to 32   | 768 default, MRL down to 256                                |
| Vector storage (float32, native dim)  | 4KB/vector                                 | 16KB/vector                    | 3KB/vector                                                  |
| Max context / sequence length         | 32,000 tokens                              | 32,000 tokens                  | 512 tokens                                                  |
| MTEB(eng, v2) mean                    | 70.70                                      | #1 multilingual leaderboard    | Strong, not top-ranked                                      |
| Architecture class                    | Dense, decoder-only (standard Qwen3)       | Dense, decoder-only (standard Qwen3) | Custom MoE, requires `trust_remote_code=True` |
| Confirmed on vLLM (one data point)    | Yes — `--runner pooling` (vLLM ≥0.8.5)     | Yes — `--runner pooling` (vLLM ≥0.8.5) | No — [issue closed, not planned](https://github.com/vllm-project/vllm/issues/15849) |
| `/v1/embeddings` support (Ollama)     | Yes, incl. `dimensions` param              | Yes, incl. `dimensions` param  | Yes, incl. `dimensions` param                               |

Sources: <https://ollama.com/library/qwen3-embedding:8b>,
<https://ollama.com/library/qwen3-embedding:0.6b>,
<https://ollama.com/library/nomic-embed-text-v2-moe>,
<https://docs.ollama.com/api/openai-compatibility>,
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/model-recipes/qwen3-embedding-8b.html>

All three are Matryoshka-trained (MRL): each accepts an optional `dimensions`
field on the request to truncate its native output to a smaller width,
trading some retrieval accuracy for a smaller vector. Dimensions support
alone isn't the deciding factor — for the original nomic-vs-8b comparison,
the native defaults already differed by more than either model's own MRL
range closes; for 0.6b specifically, architecture portability is now the
deciding factor (see §5). The vLLM row is one concrete, checkable data
point for that portability, not the requirement itself — this repo isn't
committed to vLLM as the production backend.

Note on the Ollama tag for 0.6B: it now ships as an official `library/`
tag (`qwen3-embedding:0.6b`), the same as the 8B model. It previously
existed only as a community `dengcao/` tag — if a `.env` or compose file
still references that, switch it to the official tag.

---

## 2. When to Reach for Each

**`qwen3-embedding:0.6b` — the portable default for short-to-medium text.**

- Same use cases as nomic below (chat messages, tickets, log lines, short
  facts) but with room to grow: 32K context instead of 512 tokens means
  occasional longer inputs don't silently truncate.
- Anywhere the production backend isn't locked in yet, or might change —
  a plain dense architecture is the safer bet across whichever
  OpenAI-compatible server ends up running it, not just Ollama in dev.
- 1024-dim output is a reasonable middle ground: ~3x nomic's storage cost
  per vector, but a fraction of the 8B model's 16KB/vector.

**`nomic-embed-text-v2-moe` — short-text, high-volume, cost-sensitive
retrieval, where the serving backend is fixed and known to support it.**

- Embedding chat messages, support tickets, log lines, product titles,
  short user-submitted facts — anything that comfortably fits in 512
  tokens (roughly 350-400 English words).
- A memory or search index that will accumulate a large number of vectors
  over time, where storage and index size compound (768-dim is ~3x cheaper
  per vector than 0.6b's 1024-dim, before any index overhead).
- A constrained dev/CI environment where a 958MB pull is meaningfully
  better than a larger one — laptops, CI runners, anywhere disk or
  first-pull time is a real cost.
- Still viable wherever the actual serving backend for this environment is
  confirmed to support it (Ollama does) — the gap in §5 only matters for a
  backend that hasn't been confirmed, or that's expected to change later.

**`qwen3-embedding:8b` — long-document or multilingual-quality retrieval.**

- Embedding whole documents, transcripts, or postmortems where the text
  routinely exceeds a few hundred words — 512 tokens truncates silently
  on a smaller-context model, and truncation is a correctness bug, not a
  performance one: the tail of the document is simply never indexed.
- Retrieval quality is the binding constraint and the corpus or query
  language isn't predominantly English — the #1 MTEB multilingual result
  is the one place this model is unambiguously the stronger choice.
- Storage cost or pull size isn't the bottleneck (production infrastructure
  with dedicated vector-DB capacity planning, not a laptop).

**Rule of thumb:** context length is usually the harder constraint, not
model quality. A stronger model that silently truncates half your input is
worse than a weaker model that ingests all of it. Check the input length
distribution of what's actually being embedded before optimizing for
leaderboard rank. But check deployability before either — see §5.

---

## 3. General Rules for Selecting an Embedding Model

**Match the context window to what you're actually embedding.** Measure
the token length of a representative sample of real inputs before picking
a model — not the average length, the tail. A model with a 512-token limit
silently truncates anything longer; there's usually no error, just missing
signal in the vector. If inputs regularly exceed the window, either pick a
longer-context model or chunk the text before embedding — don't let
truncation happen by accident.

**Vector width is a storage and latency cost, not just an accuracy dial.**
Every dimension is 4 bytes (float32) times however many vectors accumulate.
768 vs. 4096 dimensions is a ~5x difference in raw storage, and it also
affects index build time and search latency in most vector databases at
scale. Don't default to the largest available width "for quality" without
checking whether the retrieval task actually benefits from it — for short,
narrow-domain text, a smaller model often performs within noise of a larger
one on the metric that matters (top-k retrieval relevance), not just on a
general-purpose leaderboard.

**Confirm the vector database schema and the model output actually agree
— don't hardcode a width from a doc page.** The single most common
integration bug is a collection created with `VectorParams(size=1536)`
because that's what a tutorial for a _different_ model said, then a
mismatch error the first time a real embedding is upserted. Embed one
string and read `len(vector)` before creating the collection; never
trust a dimension number without running it.

**Keep the embedding model and the vector store schema decoupled from the
chat model.** They don't need to come from the same provider or even the
same family — an embedding model choice is a retrieval-quality and cost
decision, independent of which LLM generates responses. Wire both through
config (base URL, model name, API key), not hardcoded, the same way a chat
model is — swapping the embedding model later shouldn't require a code
change, only a config change and a full re-embed of existing data (see
below).

**Changing embedding models is not a hot-swappable operation.** Two
different models — or the same model at two different `dimensions`
settings — produce vectors that are not comparable to each other, even at
the same width. Switching models means either a new collection (and a
migration/backfill of every existing vector) or accepting that old and new
vectors can never be meaningfully compared in the same search. Plan for
this before picking a model, not after outgrowing one — decide up front
whether "point the config at a new model" is expected to be routine or
rare for this system.

**MTEB rank is a tiebreaker, not the first filter.** Leaderboard position
answers "which model is best at retrieval in general," not "which model is
best for my input shape, my latency budget, and my storage constraints."
Filter by deployability (§5), then context length and vector-width cost;
use leaderboard rank only to choose among the models that already clear
those bars.

---

## 4. Where This Applies in This Repo

`app/core/config.py`'s `embedding_base_url` / `embedding_model` /
`embedding_api_key` make the model a config value, not a hardcoded
import — switching models later is a `.env` change plus dropping and
recreating the Qdrant collection at the new width, *provided the target
model can be served by whatever backend that environment uses*. That
proviso is the point of §5: a model swap that's a pure `.env` change on
one backend can be a hard blocker on a different one if the new model's
architecture isn't supported there — this repo intentionally hasn't
committed to a single production inference server, so "which backend"
isn't a fixed answer to check against.

If the memory store's use case shifts from short facts to long documents,
that's a trigger to revisit this choice on quality/context grounds. If a
production backend is chosen or changes, that's a trigger to re-verify §5
for whatever model is currently configured, independent of whether
anything else changed.

---

## 5. Architecture Portability: a Filter Before MTEB Rank

This repo is deliberately not locked to one inference backend —
`docs/ollama-to-vllm-pattern.md`'s whole premise is a config-driven
`base_url`/`model` behind `ChatOpenAI`, so Ollama, vLLM, or any other
OpenAI-compatible server (TGI, llama.cpp's server, LM Studio, etc.) can run
underneath without a code change. **vLLM is a backend this repo supports
and has a concrete deployment sketch for — not the only backend it's
willing to run on.** The point of this section isn't "vLLM support is
mandatory," it's that a model's architecture determines how portable it is
*across* that set of backends, and that's worth checking before context
length, storage cost, or leaderboard rank, because an architecture only one
backend can load makes every other column in §1 moot the moment the
deployment target changes.

**What ruled out nomic:** `nomic-embed-text-v2-moe` uses a custom
Mixture-of-Experts architecture that requires `trust_remote_code=True` to
load. That's exactly the class of model inference servers other than the
one it was built for tend to lag on or skip — concretely, a GitHub issue
asking vLLM to support it
([vllm-project/vllm#15849](https://github.com/vllm-project/vllm/issues/15849))
was closed as not planned. It isn't a vLLM-only problem; it's a signal that
this specific architecture is narrow, and it's disqualifying for any
environment where the serving backend isn't pinned to something already
confirmed to support it.

**What makes the Qwen3-Embedding family more portable:** all three sizes
(0.6B, 4B, 8B) are dense, decoder-only models built on the standard Qwen3
architecture — no custom remote code, the same architecture family most
OpenAI-compatible servers already handle for chat. vLLM ≥0.8.5 is one
confirmed backend, serving them via:

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling
```

The `--runner pooling` flag is required and easy to miss on vLLM
specifically: every Qwen3-Embedding checkpoint declares
`architectures: ["Qwen3ForCausalLM"]` in its `config.json` — identical to
the generative Qwen3 models — so without the flag vLLM loads it as a chat
model instead of an embedding model. A different backend may have an
equivalent flag or none at all; check its docs rather than assuming this
one carries over.

**If vLLM ends up being the backend for embeddings too:** it serves one
task per running instance, so production would need a second `vllm`
service alongside the chat-model one already sketched in
`docs/ollama-to-vllm-pattern.md` — one instance can't multiplex a
generative model and a pooling model the way Ollama's daemon serves
multiple pulled models from one process. This only applies if vLLM is the
backend actually chosen; it isn't a requirement of the architecture check
itself.

**Before committing to any embedding model, regardless of which backend
ends up running it:**

1. Prefer a plain, standard architecture (dense encoder/decoder, no
   `trust_remote_code`) over a custom or MoE one — it's the trait that
   correlates with broad support, not any one server's support list.
2. If a specific backend is already a strong candidate (vLLM, given the
   existing compose sketch, or another), check whether the model's
   architecture appears in that backend's supported models list, or
   whether an open/closed GitHub issue already answers it.
3. Actually start it on that backend — a throwaway container, e.g.
   `vllm serve <model> --runner pooling` — and confirm it comes up and
   returns embeddings of the expected width. Don't trust a doc page's
   claim of support without running it, same rule as §3's vector-width
   check.
4. Only after both pass, compare context length, storage cost, and MTEB
   rank per §1–§3.
