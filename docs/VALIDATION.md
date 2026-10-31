# Validation record

31 October 2026. Local and hosted Ubuntu checks are recorded below. They verify the demonstrated behavior, not production availability or general answer accuracy.

## Environment and automated checks

Core runtime dependencies were installed from binary wheels into a new macOS arm64 CPython 3.12.2 virtual environment, separate from the original project. `pip check` passed. `requirements-lock-macos-py312.txt` records the exact resolved set, including NumPy 2.4.4, Sentence Transformers 5.4.1, PyTorch 2.11.0 and Transformers 5.5.4. The default MiniLM model revision was already cached; model execution used CPU with Hugging Face and Transformers offline modes enabled.

All **29 tests passed**: 26 unit tests and three real-model integration checks. Unit coverage includes cleaning, section labels, overlapping chunk boundaries, empty/missing corpus failures, index preconditions, cosine ranking, metadata filters, similarity thresholds, deduplication, neighbor expansion, query preservation, refusals, citation support, evaluation counts and lazy/pinned model loading. Entry-point help works without the model library being loaded. Real-model checks cover 384-dimensional embeddings, a cited answer on the fictional notes and an unsupported-query refusal.

The neighbor-citation regression was demonstrated as a failing test before correction. Previously, a sentence drawn from expanded context could retain the ID of the retrieval anchor even when that chunk did not contain the answer. Sentence selection now retains the ID and section of a supporting chunk. Retrieval scoring and answer text are unchanged by that correction.

The model loader reports an unexpected `embeddings.position_ids` checkpoint key with the pinned model/runtime. Model construction and all integration/evaluation checks still complete. That warning is recorded rather than treated as proof of failure or suppressed.

## Historical development evaluation

The untouched original source and the candidate both ran against the original 37-file local corpus and 16 development questions. All aggregate measurements matched: 1,085 chunks, 384 embedding dimensions, baseline keyword match 11/13, naive match 10/13, improved match 13/13, and improved negative rejection 3/3. See [evaluation notes](EVALUATION.md) for the metric definitions and limitations. The article text is not part of this repository.

## Platform and service limits

GitHub Actions on Ubuntu 24.04 passed four jobs: 26 unit tests on each of Python 3.12 and 3.13, plus all 29 tests in a separate real-model job on each version. There were no test failures or skips. Unit tests are included in the 29-test suite, so these are 29 distinct tests. The validated workflow is included in this public source release; [current runs](https://github.com/daley-tyler/article-retrieval-chatbot/actions/workflows/tests.yml) show its public CI status.

Integration jobs install CPU-only PyTorch 2.11.0 and the prepared runtime requirements, check dependencies, and explicitly download the pinned public MiniLM model from Hugging Face. Tests then run with offline flags and the included fictional documents. No custom secret, original corpus, Docker setup or administrator access is needed. These Ubuntu checks do not rerun the historical Wikipedia evaluation. Windows support, hosted application availability and large-corpus performance remain unverified.

The optional Wikipedia downloader and exact reconstruction of the historical 37-file corpus were not run in the clean environment. The pinned corpus manifest allows an existing local copy to be checked. The included fictional demonstration and automated tests do not require that dataset or a paid service.
