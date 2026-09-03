# Choosing an Embedding Model — qwen3-embedding vs. nomic-embed-text-v2-moe

We picked `nomic-embed-text-v2-moe` for the semantic memory store over
`qwen3-embedding:8b`. This doc is the comparison behind that call, plus the
general rules worth applying the next time an embedding model has to be
chosen — for this repo's memory store or anything else.

---

## 1. The Two Models, Side by Side

|                                      | `qwen3-embedding:8b`          | `nomic-embed-text-v2-moe`                                |
| ------------------------------------ | ----------------------------- | -------------------------------------------------------- |
| Ollama pull tag                      | `qwen3-embedding:8b`          | `nomic-embed-text-v2-moe:latest`                         |
| Model size on disk                   | 4.7GB                         | 958MB                                                    |
| Parameters                           | 7.57B (dense)                 | 475M total / 305M active (MoE, 8 experts, top-2 routing) |
| Output dimensions                    | 4096 default, MRL down to 32  | 768 default, MRL down to 256                             |
| Vector storage (float32, native dim) | 16KB/vector                   | 3KB/vector                                               |
| Max context / sequence length        | 32,000 tokens                 | 512 tokens                                               |
| MTEB ranking                         | #1 multilingual leaderboard   | Strong, not top-ranked                                   |
| `/v1/embeddings` support (Ollama)    | Yes, incl. `dimensions` param | Yes, incl. `dimensions` param                            |

Sources: <https://ollama.com/library/qwen3-embedding:8b>,
<https://ollama.com/library/nomic-embed-text-v2-moe>,
<https://docs.ollama.com/api/openai-compatibility>

Both are Matryoshka-trained (MRL): each accepts an optional `dimensions`
field on the request to truncate its native output to a smaller width,
trading some retrieval accuracy for a smaller vector. Neither model's
`dimensions` support is the deciding factor below — the native defaults
already differ by more than either model's own MRL range closes.

---

## 2. When to Reach for Each

**`nomic-embed-text-v2-moe` — short-text, high-volume, cost-sensitive
retrieval.**

- Embedding chat messages, support tickets, log lines, product titles,
  short user-submitted facts — anything that comfortably fits in 512
  tokens (roughly 350-400 English words).
- A memory or search index that will accumulate a large number of vectors
  over time, where storage and index size compound (768-dim is ~5x
  cheaper per vector than 4096-dim, before any index overhead).
- A constrained dev/CI environment where a 958MB pull is meaningfully
  better than a 4.7GB one — laptops, CI runners, anywhere disk or
  first-pull time is a real cost.
- Lab 38's case exactly: `remember()` embeds one short fact per call, and
  the store is meant to grow without becoming its own capacity problem.

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
leaderboard rank.

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
Filter by context length and vector-width cost first; use leaderboard rank
to choose among the models that already clear those bars.

---

## 4. Where This Applies in This Repo

`app/core/config.py`'s `embedding_base_url` / `embedding_model` /
`embedding_api_key` (Lab 38) make the model a config value, not a hardcoded
import — switching from `nomic-embed-text-v2-moe` to `qwen3-embedding:8b`
later is a `.env` change plus dropping and recreating the Qdrant collection
at the new width (see Lab 38's Safety Net / Rollback), not a code change.
If the memory store's use case shifts from short facts to long documents,
that's the trigger to revisit this choice — not a leaderboard update.
