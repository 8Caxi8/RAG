*This project has been created as part of the 42 curriculum by dansimoe.*
 
# RAG against the machine
 
## Description
 
This project implements a **Retrieval-Augmented Generation (RAG) system** that answers
questions about the [vLLM](https://github.com/vllm-project/vllm) codebase. It ingests the
vLLM source repository (Python code and Markdown documentation), builds a searchable BM25
index, retrieves the most relevant chunks for a given question, and generates a
natural-language answer grounded in that retrieved context using **Qwen/Qwen3-0.6B**.
 
The system is exposed as a command-line tool (`src`, built with Python Fire) with six
commands covering the whole pipeline: `index`, `search`, `search_dataset`, `answer`,
`answer_dataset`, and `evaluate`.
 
## Instructions
 
### Requirements
 
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) as the package/project manager
- The `vllm-0.10.1` repository and the provided datasets, unzipped locally (see
  [Example usage](#example-usage) for the expected paths)
### Installation
 
```bash
make install
```
 
This installs `uv` if it isn't already available, and syncs all project dependencies
(`pyproject.toml` / `uv.lock`).
 
### Running
 
```bash
make run ARGS="<command> [options]"
```
 
See [Example usage](#example-usage) below for concrete commands, or run:
 
```bash
make run ARGS="--help"
```
 
### Linting
 
```bash
make lint
```
 
Runs `flake8` and `mypy` with the mandatory flags specified by the subject. (An optional
`lint-strict` target running `mypy --strict` was considered but ultimately dropped — see
[Design decisions](#design-decisions).)
 
## System architecture
 
The pipeline has five components, each in its own module:
 
```
                 ┌──────────────┐
  repo files ──▶ │ chunking.py  │──▶ list[Chunk]
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   bm25.py    │  BM25Index: build / search / save / load
                 └──────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       retrieved Chunks     ┌──────────────┐
              │             │    llm.py    │  BaseLLM / Qwen3LLM
              ▼             └──────────────┘
       ┌──────────────┐            │
       │ evaluate.py  │            ▼
       │  (recall@k)  │      generated answer
       └──────────────┘
              ▲                   │
              └─────────┬─────────┘
                         ▼
                 ┌──────────────┐
                 │ __main__.py  │  Fire CLI: index / search / search_dataset /
                 │              │  answer / answer_dataset / evaluate
                 └──────────────┘
```
 
- **`chunking.py`** — turns raw files into `Chunk` objects (`file_path`, `text`,
  `first_character_index`, `last_character_index`).
- **`bm25.py`** — tokenizes chunks, fits a BM25 model (via `bm25s`), and answers
  top-k queries. Persists to disk so `index` and `search` can run as separate
  invocations.
- **`llm.py`** — an abstract `BaseLLM` (model/tokenizer loading, `generate()`,
  token-budget truncation) with a `Qwen3LLM` implementation. Designed so a second
  model class can be dropped in later by only implementing `build_prompt()`.
- **`evaluate.py`** — recall@k against ground-truth sources, with a
  configurable character-overlap threshold (default 5%, per the subject).
- **`models.py`** — the pydantic models specified by the subject
  (`MinimalSource`, `AnsweredQuestion`, `StudentSearchResults`, ...), used for
  all JSON input/output.
- **`__main__.py`** — the Fire CLI wiring everything together, plus context
  assembly for the LLM prompts.
## Chunking strategy
 
Two strategies, chosen by file extension:
 
- **Python (`.py`)** — parses the file with the `ast` module and creates one chunk per
  top-level statement (function, class, import, ...), so a chunk is never cut in the
  middle of a function. If a node (e.g. a large class) exceeds `max_chunk_size`, it is
  recursively split along its children (e.g. one chunk per method) before falling back to
  a hard character split as a last resort. Files that fail to parse (syntax errors) fall
  back to the text strategy below.
- **Markdown / text (`.md` and others)** — splits on blank-line paragraphs, then greedily
  packs consecutive paragraphs into a chunk until the next one would exceed
  `max_chunk_size`. Consecutive chunks overlap by a configurable number of characters
  (default 200) so a fact split across a chunk boundary is still retrievable from either
  chunk.
`max_chunk_size` defaults to 2000 characters and is configurable via `--max_chunk_size`
on the `index` command, as required by the subject.
 
Character offsets are computed exactly (verified in testing against the original file
content byte-for-byte), which is what makes `first_character_index` /
`last_character_index` reliable for both retrieval and re-reading source text later (in
`answer_dataset`).
 
## Retrieval method
 
**BM25**, via the [`bm25s`](https://github.com/xhluca/bm25s) library rather than
`rank_bm25` — see [Challenges faced](#challenges-faced) for why.
 
- **Tokenization**: lowercased, split on non-alphanumeric characters; identifiers in
  `snake_case` or `camelCase` are additionally split into their sub-words (so a query
  like "openai server" also matches a symbol like `OpenAIServer`); common English
  stopwords are removed from both the indexed chunks and the query.
- **Hyperparameters**: `k1=1.5` (default), `b=0.2` (tuned down from the library default
  of `0.75` — see [Performance analysis](#performance-analysis)), using the classic
  Robertson IDF formula (`method="robertson"` in `bm25s`) to match `rank_bm25`'s formula.
- **Ranking**: top-k by BM25 score. Scores can legitimately be negative on this corpus
  (see Challenges faced), so ranking is always by relative rank, never filtered by score
  sign.
## Performance analysis
 
Measured on the full `vllm-0.10.1` repository (71,167 chunks after indexing), on CPU
(no working CUDA driver available in the test environment):
 
| Metric | Required | Measured |
|---|---|---|
| Indexing time | ≤ 5 min | **9.1s** |
| Cold start latency (first search) | ≤ 60s | **7.84s** |
| Warm retrieval throughput (1000 questions) | ≤ 90s | **~0.8s** (projected from 100 questions) |
| Recall@5, docs questions | ≥ 80% | **0.810** |
| Recall@5, code questions | ≥ 50% | **0.550** |
 
Full recall curve (`dataset_docs_public.json` / `dataset_code_public.json`, 100 questions
each, k=10 retrieved):
 
| | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| docs | 0.530 | 0.740 | 0.810 | 0.880 |
| code | 0.340 | 0.470 | 0.550 | 0.640 |
 
**Answer quality** (manual review of ~15 random questions per dataset, after fixing the
two formatting issues described in Challenges faced):
 
- **docs**: ~80% correct or essentially correct, ~7% partial, ~13% wrong.
- **code**: ~60% correct, ~20% partial, ~20% wrong.
Code questions retrieve more reliably (function/class/parameter names are strong lexical
anchors for BM25) but answer generation is less forgiving — code questions tend to ask for
one exact fact (a default value, an exact exception message), leaving no room for a
"close enough" answer the way a docs question does.
 
## Design decisions
 
- **`bm25s` over `rank_bm25`**: see Challenges faced — throughput requirement.
- **Character-based context pre-filter + token-based hard cap**: chunks are first
  greedily packed by character count (cheap, no model needed) up to
  `--max_context_length`, then the assembled context is truncated to
  `--max_context_tokens` using the LLM's own tokenizer right before generation. This
  satisfies the subject's requirement to pass context "within token limits" precisely,
  while keeping the cheap heuristic for picking *which* chunks to include.
- **Pydantic only where it earns its keep**: the subject requires "all classes must use
  pydantic". `Chunk` (in `chunking.py`) was converted to a `pydantic.BaseModel` since it
  is pure data crossing an internal boundary (chunking → indexing) and gets real
  validation for free (non-negative offsets, `last >= first`). `BaseLLM`/`Qwen3LLM`
  (wrapping a `torch.nn.Module` and tokenizer) and `BM25Index` (wrapping a `bm25s.BM25`
  sparse-matrix model) were kept as plain classes: they wrap stateful third-party objects
  with no meaningful "data" to validate, and forcing them into `BaseModel` would need
  `arbitrary_types_allowed=True` for zero actual type-safety benefit.
  `RagCLI` is a Fire entrypoint (behavior, not data) and was left as-is for the same
  reason.
- **`lint-strict` dropped from the Makefile**: `mypy --strict` flags a handful of
  unavoidable issues in third-party libraries with no published type stubs (`bm25s`,
  `fire`) and a couple of untyped calls inside `transformers` itself
  (`logging.set_verbosity_error()`, `model.eval()`). These aren't fixable from this
  project's code without fighting the type checker for no real safety gain, and the
  subject marks `lint-strict` as optional, so it was removed rather than kept in a
  permanently-failing state. `make lint` (the mandatory target) is clean.
- **Greedy decoding (`do_sample=False`)** for answer generation, for reproducibility —
  appropriate for a system meant to be faithful to retrieved context rather than
  creative.
## Challenges faced
 
- **A negative-score bug that silently dropped valid retrieval results.** The BM25 IDF
  formula can go negative when a term appears in more than half the corpus (common on a
  small or homogeneous corpus). An early version of `search()` filtered `scores[i] > 0`
  before taking the top-k, which meant that whenever *all* candidate scores were
  negative, `search()` returned nothing at all — even for an exact, obviously relevant
  match. Fixed by always ranking by relative score, never filtering by sign.
- **`rank_bm25` was too slow for the throughput requirement.** On the full corpus
  (71k chunks), `rank_bm25`'s pure-Python `get_scores()` projected to ~132s for 1000
  questions — over the 90s budget. Switched to `bm25s` (vectorized, sparse-matrix
  implementation), which brought this down to ~0.8s projected — but its default IDF
  variant (`method="lucene"`) is not the classic Robertson formula, and gave slightly
  different (and, on this corpus, slightly worse) rankings than the ones the `b`
  parameter had been tuned against. Explicitly setting `method="robertson"` reproduced
  the original rankings and recall scores exactly.
- **A large retrieved context made CPU generation extremely slow.** With `k=10` and
  `max_chunk_size=2000`, `answer_dataset` could receive up to ~20,000 characters of
  context per question — the CPU prefill cost of that prompt dominated generation time
  (~22s/question). Capping context (first by character budget, then by an actual token
  budget via the tokenizer) brought this down substantially, without touching retrieval
  quality (retrieval still uses the full `k`; only what's shown to the LLM is capped).
- **The LLM occasionally hallucinated a URL by pattern-matching a neighboring one.**
  Asked to name a contributing-guide's source file, the model correctly cited a real
  HuggingFace URL from the context, but then invented an analogous
  `https://github.com/vllm-project/vllm/blob/main/src/...` URL for a *different* link in
  the same context block that used a different (non-URL) reference scheme — copying the
  `src/`-prefixed path structure from its neighbor, which vLLM's repository does not
  actually use. Tightening the system prompt to forbid inventing URLs and to only cite a
  path if it appears verbatim in the context reduced this significantly, though this kind
  of pattern-contamination between nearby context blocks is a real, residual limitation
  of a 0.6B model that a stronger model would likely handle better.
  A related formatting bug — the model sometimes copied the literal `# Source: <path>`
  context-block marker into its answer — was fixed by rewording the prompt to describe
  that marker as formatting metadata, not something to repeat.
  Both issues were only caught by manually sampling answers rather than looking at recall
  numbers alone (recall measures retrieval, not generation faithfulness).
- **BM25 has no semantic understanding**, which shows up as confident wrong answers, not
  just missing ones. E.g. a question about *"data parallel deployment"* retrieved
  `expert_parallel_deployment.md` ahead of the actually-relevant
  `data_parallel_deployment.md`, purely on lexical overlap (both pages share many
  "parallel", "deployment", "configure" terms) — the LLM then answered about Expert
  Parallelism instead. This is a known limitation of lexical retrieval; the subject's
  bonus section (semantic embeddings, hybrid retrieval) exists precisely to address it.
- **A stray `pyproject.toml` setting silently defeated the point of `make lint`.**
  `[tool.mypy] strict = true` in `pyproject.toml` is picked up automatically by `mypy`
  regardless of which CLI flags are passed, since CLI flags only override the specific
  options they name — `strict = true` still enabled checks the mandatory `lint` flags
  never asked for. This meant `make lint` was silently running closer to `--strict` than
  intended. Removed from `pyproject.toml`, keeping the strict/mandatory distinction
  purely in the Makefile's flags.
## Example usage
 
Assumes `vllm-0.10.1/` and the datasets are unzipped under `data/`:
 
```
data/
├── raw/vllm-0.10.1/                      # the vLLM repository
└── datasets_public/public/
    ├── AnsweredQuestions/
    │   ├── dataset_docs_public.json
    │   └── dataset_code_public.json
    └── UnansweredQuestions/
        ├── dataset_docs_public.json
        └── dataset_code_public.json
```
 
Build the index:
 
```bash
make run ARGS="index --repo_path data/raw/vllm-0.10.1 --b 0.2"
```
 
Search for a single query:
 
```bash
make run ARGS="search 'How to configure the OpenAI server?' --k 5"
```
 
Search a whole dataset and save results:
 
```bash
make run ARGS="search_dataset --dataset_path data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results"
```
 
Evaluate retrieval quality against ground truth:
 
```bash
make run ARGS="evaluate --student_search_results_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
```
 
Answer a single question:
 
```bash
make run ARGS="answer 'What is Retrieval-Augmented Generation?' --k 5"
```
 
Generate answers for a whole dataset:
 
```bash
make run ARGS="answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer"
```
 
## Resources
 
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and
  Beyond.* — the original BM25 formulation used here.
- [`bm25s`](https://github.com/xhluca/bm25s) — the retrieval library used, and its
  accompanying paper (Lù, 2024, *bm25s: Orders of Magnitude Faster Lexical Search via
  Eager Sparse Scoring*).
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks.* — the original RAG paper.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) (Hugging Face).
- [pydantic documentation](https://docs.pydantic.dev/) — data models and validation.
- [Python `ast` module documentation](https://docs.python.org/3/library/ast.html) — used
  for syntax-aware Python chunking.
- [vLLM documentation](https://docs.vllm.ai/) — the corpus being indexed and queried.
### AI usage
 
AI was used throughout this project's implementation, under the
supervision and review described in Chapter II of the subject. Specifically:
 
- **Debugging**: diagnosing concrete bugs found while testing against the real vLLM
  corpus and datasets — the BM25 negative-score filtering bug, the `rank_bm25` →
  `bm25s` IDF-formula mismatch, the context-truncation/throughput issue.
- **Performance tuning**: guided experimentation with BM25's `b`/`k1` parameters and
  context-length limits, each change measured against real recall/timing numbers before
  being kept.

All AI-assisted code was read, tested, and understood before being kept — per Chapter
II's guidance, no code was included that could not be explained during a defense.