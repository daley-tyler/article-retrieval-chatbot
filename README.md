# Article Retrieval Chatbot

I built this university project to explore how chunking, embeddings and retrieval rules affect answers over a small collection of articles. It searches local text files and returns a sentence from the retrieved context with a source citation. If the context does not support an answer, it can refuse the question.

The answers are extractive. The code builds a grounded prompt for inspection, but it does not call a generative LLM or an external answering API. The included demonstration uses fictional service notes that I can distribute with the source. This repository is the public source snapshot; the original coursework and development history remain preserved separately.

## How it works

```text
Local .txt files → cleaning and sentence chunks → MiniLM embeddings
                                              ↓
Question → original and keyword queries → NumPy cosine search
                                              ↓
                 title/keyword reranking, filtering and deduplication
                                              ↓
                 neighboring context → sentence selection or refusal
```

The default encoder is `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. Normalized vectors are held in memory and searched with NumPy dot products. There is no vector database or saved index; each run rebuilds the index. Default chunks contain three sentences with one sentence of overlap. Retrieval considers 12 candidates and returns up to four after scoring and filtering.

The code keeps titles, section labels, source paths and chunk IDs. Nearby chunks can supply more context, and the final citation identifies a chunk containing the selected sentence. Query rewriting uses simple keyword processing. The refusal rules combine similarity, topic anchors and sentence overlap; they are heuristics rather than a guarantee of correctness.

## Setup and demonstration

Use CPython 3.12 or 3.13. The full test suite passes on Ubuntu 24.04 with both versions and on macOS arm64 with Python 3.12.2. To reproduce the macOS environment, run from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock-macos-py312.txt
python demo.py
```

The first demonstration run downloads the pinned public embedding model from Hugging Face. No API key is needed. Documents and questions are embedded locally on CPU; this project does not send their text to an inference service. To require an already cached model and avoid model downloads:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python demo.py --local-files-only
```

The supported example asks where the fictional Harbor service stores deployment records. It returns the sentence naming its SQLite database, with a `Harbor.txt#chunk-...` citation. A question about Apollo 11 is refused because those notes do not contain its answer. Start interactive mode with:

```sh
python rag_chatbot.py --documents examples/documents
```

Use `debug:` before a question to inspect retrieved context and scores. To search your own text files, replace `examples/documents` with a local directory. Do not add private documents, access tokens, model weights or local result files to Git.

`requirements.txt` pins the direct runtime dependencies for other environments. On Ubuntu, the verified CPU setup is:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: -r requirements.txt
python demo.py
```

Use `python3.13` instead for Python 3.13. The full lock file records the resolved macOS environment; it is not a Linux or Python 3.13 runtime lock. [GitHub Actions](https://github.com/daley-tyler/article-retrieval-chatbot/actions/workflows/tests.yml) runs unit and real-model integration checks on Ubuntu 24.04 with both Python versions.

## Tests

The unit tests need only NumPy and use an injected deterministic encoder. They check text processing, chunk boundaries, ranking, metadata filtering, context expansion, duplicate removal, refusals, citations and model-loading boundaries without downloading weights:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p 'test_retrieval.py' -v
```

With the full runtime installed and MiniLM cached, run the unit tests plus three real-model integration checks:

```sh
RAG_RUN_MODEL_TESTS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

All **29 tests pass** in the verified environments: 26 unit tests and three real-model integration checks. The integration checks cover embedding dimensions, a cited answer and an unsupported question. CI downloads the pinned public MiniLM model from Hugging Face without a token, then runs the tests offline against the fictional documents. Synthetic test results do not measure general retrieval quality. [Validation](docs/VALIDATION.md) records the environments and checks actually completed.

## Development evaluation

The original evaluation uses 37 Wikipedia article files and 16 hand-selected questions: 13 supported and three unsupported. Both the original source and this preparation reproduced the following aggregate results locally:

| Measure | Article baseline | Naive retrieval | Improved retrieval |
| --- | --- | --- | --- |
| Supported-answer keyword match | 11/13 | 10/13 | 13/13 |
| Unsupported-question refusal | 0/3 | 0/3 | 3/3 |

These are small development-set checks, not a held-out benchmark. A keyword appearing in an answer does not establish factual accuracy. The original “context relevance” measure duplicates document hit rate, while “fidelity” checks source containment or refusal and assigns the article baseline a value of one by construction. [Evaluation notes](docs/EVALUATION.md) explain those limits and how to reproduce the checks when the historical corpus is available.

The Wikipedia text is excluded. `docs/corpus-manifest.json` records its filenames and hashes, and `evaluation_questions.json` retains the project questions. The optional downloader uses a pinned dataset revision but is not verified to reconstruct the exact historical 37-file selection. The project runs and tests with the included fictional documents without that download.

## Limits and provenance

This is a small CLI prototype. Cleaning and sentence/section detection are heuristic and English-oriented. It supports plain text, not PDF parsing, OCR or uploads. Full-matrix retrieval and repeated sentence encoding suit small collections; large-scale indexing, authentication, monitoring and production availability are outside its scope. Refusals can miss valid answers, and plausible-looking citations do not certify a document's truth or authority.

The project grew from my Generative AI coursework. I used Codex for this later reliability pass, tests and documentation. [Provenance](docs/PROVENANCE.md) distinguishes the original project, course context, preparation changes and third-party dependencies. The original history and coursework remain preserved separately. No source-code reuse license has been assigned, and no license grant for excluded material is implied. Model and dataset terms are linked in [data and model notes](docs/DATA_AND_MODEL.md).
