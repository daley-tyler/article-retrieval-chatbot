# Evaluation and reproduction

The original development corpus consists of 37 local `.txt` article files. Both original source and prepared source were evaluated on those files using the existing 13 supported questions and three unsupported questions. Both runs produced 1,085 chunks with 384-dimensional embeddings and identical aggregate results. The prepared runtime used pinned MiniLM weights on CPU in a clean macOS Python 3.12 environment.

The improved system matches expected keywords on 13 of 13 supported questions and refuses all three unsupported questions. The article baseline matches 11 of 13 and refuses none; naive retrieval matches 10 of 13 and refuses none. Naive and improved retrieval both place the expected document first for all 13 supported questions. These numbers describe this small development set only. The questions and thresholds were explored during development; there is no independent blind test, human factual assessment or evidence of broad accuracy.

## What the metrics mean

`answer_accuracy` is a legacy JSON field name. It tests whether any supplied gold keyword is a substring of the answer. It can reward an answer that mentions the expected term while making a different claim.

Retrieval top-1 checks the first returned document name. Top-k checks whether the expected document occurs anywhere in the returned context. The legacy `context_relevance` field repeats top-k document presence, so it is not an independent context-quality measure.

The legacy `answer_fidelity` field checks whether the answer is contained in retrieved context or is the fixed refusal message. The article baseline receives a value of one unconditionally for supported questions. Containment is not evidence of semantic entailment or factual truth. The CLI uses more literal labels and prints these qualifications rather than promoting them as independent quality scores.

Refusal rate counts the exact fixed refusal message on the three negative questions. It says nothing about the range of unsupported questions that might arise in practice. Empty supported/negative categories in the original evaluator use fallback denominators; only the recorded nonempty categories should be interpreted here.

## Reproduce with the historical corpus

The article text is excluded. If you already have an authorized local copy, check it before comparing results:

```sh
python tools/check_corpus.py /path/to/article_txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluate_rag.py \
  --documents /path/to/article_txt --local-files-only
```

The manifest records all 37 names and SHA-256 hashes. Changing files, model revision, preprocessing or thresholds creates a different experiment. The optional downloader selects 40 rows of a pinned Wikipedia dataset snapshot; it has not been verified to produce the exact 37-file historical selection. Do not claim an equivalent benchmark unless the manifest matches.

For an independent setup check using only this repository's fictional documents:

```sh
python evaluate_rag.py --documents examples/documents --questions examples/questions.json
```

Those two synthetic examples demonstrate the workflow and are not a substitute benchmark. No original report draft, extracted article passages, embeddings or per-question answer dump is distributed as a result artifact.
