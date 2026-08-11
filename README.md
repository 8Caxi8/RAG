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
`answer_dataset`, and `evaluate`. Retrieval quality was validated against the official
`moulinette` evaluator, not just this project's own `evaluate` command — see
[Challenges faced](#challenges-faced).
 
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
 
The pipeline is split into focused modules, each with a single responsibility:
 
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
                        ▼
                 retrieved Chunks
                        │
                        ▼
                 ┌──────────────┐
                 │  context.py  │  ContextCreator: character-budgeted context
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │    llm.py    │  BaseLLM / Qwen3LLM: token-budget cap + generate()
                 └──────────────┘
                        │
                        ▼
                  generated answer
 
  ┌──────────────┐         ┌──────────────┐
  │ evaluate.py  │◀────────│  ragcli.py   │  RagCLI: wires everything together
  │  (recall@k)  │         └──────────────┘
  └──────────────┘                │
                                   ▼
                          ┌──────────────┐
                          │ __main__.py  │  thin entry point:
                          │              │  fire.Fire(RagCLI(SYSTEM_PROMPT))
                          └──────────────┘
```
 
- **`chunking.py`** — turns raw files into `Chunk` objects (`file_path`, `text`,
  `first_character_index`, `last_character_index`).
- **`bm25.py`** — tokenizes chunks, fits a BM25 model (via `bm25s`), and answers
  top-k queries. Persists to disk so `index` and `search` can run as separate
  invocations.
- **`context.py`** — `ContextCreator`, which assembles a character-budgeted context
  block from retrieved chunks or sources, greedily packing whole blocks and
  hard-truncating only as a last resort.
- **`llm.py`** — an abstract `BaseLLM` (model/tokenizer loading, `generate()`,
  token-budget truncation) with a `Qwen3LLM` implementation. Designed so a second
  model class can be dropped in later by only implementing `build_prompt()`.
- **`evaluate.py`** — recall@k against ground-truth sources, using Intersection over
  Union (IoU) to decide whether a retrieved source "found" a ground-truth one — see
  [Challenges faced](#challenges-faced) for why this matters.
- **`models.py`** — the pydantic models specified by the subject
  (`MinimalSource`, `AnsweredQuestion`, `StudentSearchResults`, ...), used for
  all JSON input/output.
- **`ragcli.py`** — the `RagCLI` class wiring everything together: the six CLI
  commands, plus small helpers for loading/saving JSON and converting between
  internal (`Chunk`) and public (`MinimalSource`) representations.
- **`__main__.py`** — a thin entry point that builds the system prompt and hands
  an already-constructed `RagCLI` instance to `fire.Fire()`.
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
  `max_chunk_size`. An optional overlap (`overlap`, in characters) can repeat the tail of
  one chunk at the start of the next, but defaults to **0** — see
  [Challenges faced](#challenges-faced) for the empirical reason.
`max_chunk_size` defaults to 2000 characters and is configurable via `--max_chunk_size`
on the `index` command, as required by the subject.
 
Character offsets are computed exactly (verified in testing against the original file
content byte-for-byte, and against a 200-scenario randomized stress test after a
boundary-overflow bug fix — see Challenges faced), which is what makes
`first_character_index` / `last_character_index` reliable for both retrieval and
re-reading source text later (in `answer_dataset`).
 
## Retrieval method
 
**BM25**, via the [`bm25s`](https://github.com/xhluca/bm25s) library rather than
`rank_bm25` — see [Challenges faced](#challenges-faced) for why.
 
- **Tokenization**: lowercased, split on non-alphanumeric characters; identifiers in
  `snake_case` or `camelCase` are additionally split into their sub-words (so a query
  like "openai server" also matches a symbol like `OpenAIServer`); common English
  stopwords are removed from both the indexed chunks and the query.
- **Hyperparameters**: `k1=1.5` (default), `b=0.2` (tuned down from the library default
  of `0.75`; re-validated after switching `overlap` to 0 — both `b=0.15` and `b=0.3` were
  re-tested against the official evaluator at that point, and `b=0.2` remained best on
  both datasets), using the classic Robertson IDF formula (`method="robertson"` in
  `bm25s`) to match `rank_bm25`'s formula. Kept as the *default* only on `RagCLI.index()`
  — `BM25Index`'s own constructor default stays at the library's generic `0.75`, since
  that class has no business being opinionated about one specific corpus (see
  [Design decisions](#design-decisions)).
- **Ranking**: top-k by BM25 score. Scores can legitimately be negative on this corpus
  (see Challenges faced), so ranking is always by relative rank, never filtered by score
  sign.
## Performance analysis
 
Measured on the full `vllm-0.10.1` repository, on CPU (no working CUDA driver available
in the test environment). All recall numbers below are from the **official `moulinette`
evaluator**, not this project's own `evaluate` command (the two agree closely, but the
official tool is authoritative — see Challenges faced):
 
| Metric | Required | Measured |
|---|---|---|
| Indexing time | ≤ 5 min | **9.1s** |
| Cold start latency (first search) | ≤ 60s | **7.84s** |
| Warm retrieval throughput (1000 questions) | ≤ 90s | **~0.8s** (projected from 100 questions) |
| Recall@5, docs questions | ≥ 80% | **0.830** |
| Recall@5, code questions | ≥ 50% | **0.530** |
 
Full recall curve (`dataset_docs_public.json` / `dataset_code_public.json`, 100 questions
each, `k=10` retrieved, final config `b=0.2`, `overlap=0`):
 
| | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| docs | 0.570 | 0.760 | 0.830 | 0.880 |
| code | 0.340 | 0.460 | 0.530 | 0.630 |
 
**Failure analysis** (via `moulinette list_valid_questions` on the docs dataset, 86/100
valid): the 14 misses cluster into recognizable patterns rather than being random —
roughly a third are narrow build/hardware facts (exact CUDA compiler versions, supported
CUDA architectures for a specific kernel) that share heavy vocabulary with many other
CUDA-related pages, diluting BM25's ability to discriminate between them; another third
are "which model architectures support X" questions, whose answers likely live in a
compatibility table (sparse prose, hard to chunk/match well); the rest are one-off misses
without an obvious shared cause. This points to a structural limit of lexical retrieval
on table-like or highly similar technical content, not an implementation bug.
 
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
- **`overlap=0` by default in `chunk_text`**: chosen empirically, not by default library
  convention — see Challenges faced.
- **Character-based context pre-filter + token-based hard cap**: chunks are first
  greedily packed by character count (cheap, no model needed) up to
  `--max_context_length`, then the assembled context is truncated to
  `--max_context_tokens` using the LLM's own tokenizer right before generation. This
  satisfies the subject's requirement to pass context "within token limits" precisely,
  while keeping the cheap heuristic for picking *which* chunks to include.
- **`ContextCreator` holds its budget as instance state**: `max_context_length` is set
  once in `__init__` rather than threaded through every method call. An earlier version
  used `@staticmethod`s that tried to call each other via `self`, which doesn't exist on
  a staticmethod — moving the shared configuration into real instance state was both the
  fix and a cleaner design.
- **`RagCLI` takes `system_prompt` as a constructor argument**, and `__main__.py` passes
  `fire.Fire()` an *already-constructed* `RagCLI(SYSTEM_PROMPT)` instance rather than the
  bare class. If Fire were given the class instead, it would treat `system_prompt` as a
  required CLI argument that must be supplied *before* the subcommand name, breaking the
  `python -m student <command> ...` invocation pattern the subject expects.
- **Pydantic only where it earns its keep**: the subject requires "all classes must use
  pydantic". `Chunk` (in `chunking.py`) was converted to a `pydantic.BaseModel` since it
  is pure data crossing an internal boundary (chunking → indexing) and gets real
  validation for free (non-negative offsets, `last >= first`). `BaseLLM`/`Qwen3LLM`
  (wrapping a `torch.nn.Module` and tokenizer), `BM25Index` (wrapping a `bm25s.BM25`
  sparse-matrix model), and `ContextCreator` were kept as plain classes: they wrap
  stateful third-party objects or pure behavior with no meaningful "data" to validate,
  and forcing them into `BaseModel` would need `arbitrary_types_allowed=True` for zero
  actual type-safety benefit. `RagCLI` is a Fire entrypoint (behavior, not data) and was
  left as-is for the same reason.
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
- **`rank_bm25` was too slow for the throughput requirement.** On the full corpus,
  `rank_bm25`'s pure-Python `get_scores()` projected to ~132s for 1000 questions — over
  the 90s budget. Switched to `bm25s` (vectorized, sparse-matrix implementation), which
  brought this down to ~0.8s projected — but its default IDF variant (`method="lucene"`)
  is not the classic Robertson formula, and gave slightly different (and, on this corpus,
  slightly worse) rankings than the ones the `b` parameter had been tuned against.
  Explicitly setting `method="robertson"` reproduced the original rankings and recall
  scores exactly.
- **This project's own `evaluate` used the wrong overlap formula.** It computed
  `overlap_length / correct_source_length` — how much of the *ground-truth* span the
  retrieved chunk covers. The official `moulinette` evaluator instead uses **true
  Intersection over Union** (`overlap / (retrieved_len + correct_len - overlap)`), which
  is symmetric and penalizes a retrieved chunk that is much larger than the correct span,
  even if it fully contains it. On a large chunk (e.g. 2000 characters) that happens to
  contain a small ground-truth span (e.g. 50 characters), the two formulas disagree
  enormously — 1.0 ("found") vs. 0.025 (not found, well under the 5% threshold). Since
  this project's chunks are often large (whole functions/classes), this was not a
  cosmetic difference. Fixed by rewriting `_overlap_ratio` (renamed `_iou`) to compute
  real IoU, matching the evaluator exactly.
- **The official evaluator rejected this project's output outright** with 100
  `pydantic.ValidationError`s (`question_str: Field required`). The subject's own PDF
  models this project was built against use the field name `question`, but the real
  `moulinette` expects `question_str` for *student-submitted* results specifically (its
  ground-truth-dataset format still uses `question` — the asymmetry is intentional on
  their end). Fixed with a `pydantic` field alias
  (`question: str = Field(alias="question_str")`, `populate_by_name=True`) so the
  Python-facing name stays `question` (matching the PDF, and not touching the rest of the
  codebase) while JSON serialization uses `question_str` — but only when explicitly
  serializing `by_alias=True`, which had to be added to every call site producing
  student-facing output.
- **The evaluator also caught a real chunk-size overflow bug**: `Source ... has a length
  of 2038 which is more than the limit of 2000 characters`. In `chunk_text`'s paragraph
  packer, when starting a new chunk after an overflow, the overlap-based start position
  was computed and used to set the new chunk's end without ever re-checking that the
  resulting window still fit within `max_chunk_size` — a large enough paragraph combined
  with the overlap recovery could push a chunk over budget. Fixed by clamping the
  recovered start position so the new chunk can never exceed `max_chunk_size` on its own,
  verified against the original failing case and a 200-scenario randomized stress test.
- **Overlap turned out to hurt more than it helped, once measured against the real
  evaluator.** The original design gave adjacent Markdown chunks a 200-character overlap
  to avoid losing a fact split across a boundary. Measured against `moulinette` (after
  the two bugs above were fixed), reducing overlap monotonically *improved* Recall@5 on
  the docs dataset (0.800 at overlap=200, 0.810 at 50, 0.830 at 0), while the code dataset
  stayed essentially flat. The likely reason: overlap duplicates content across
  neighboring chunks, and on this corpus that redundancy cost more top-k "slots" to
  near-duplicate chunks than the boundary-protection saved. `b` (BM25's length-
  normalization parameter) was re-swept at overlap=0 to confirm `0.2` was still optimal
  rather than an artifact of the old overlap setting — it was. A reminder that even a
  single shared BM25 index means a change to one file type's chunking (Markdown) can
  measurably shift scores for the *other* file type (Python) too, since IDF statistics
  are computed over the whole corpus: the code dataset's recall changed slightly even
  though no `.py` chunk was touched by the overlap change.
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
- **A large retrieved context made CPU generation extremely slow.** With `k=10` and
  `max_chunk_size=2000`, `answer_dataset` could receive up to ~20,000 characters of
  context per question — the CPU prefill cost of that prompt dominated generation time
  (~22s/question). Capping context (first by character budget, then by an actual token
  budget via the tokenizer) brought this down substantially, without touching retrieval
  quality (retrieval still uses the full `k`; only what's shown to the LLM is capped).
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
make run ARGS="index --repo_path data/raw/vllm-0.10.1"
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
- [pydantic documentation](https://docs.pydantic.dev/) — data models, validation, and
  field aliasing.
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