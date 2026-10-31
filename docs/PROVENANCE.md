# Project provenance

I developed this as an article-question-answering project for a university Generative AI course. My saved proposal lists Tyler Daley as the sole team member. The local development repository has nine commits under my historical student identity, from empty project files through the article corpus, initial retrieval implementation and later evaluation improvements. No other commit author or named project teammate was identified in those records. Git metadata alone cannot establish the origin of every line of code.

The source used for this preparation is local commit `8160b85bfc3f72c60970047a69c97b977e1238ff`. The main implementation was stored as `RAG_chatbot.py`; scripts imported `rag_chatbot`, which worked on the original case-insensitive filesystem. The prepared source consistently uses lowercase naming. The original project and its history have not been rewritten.

The course supplied the project context and taught embeddings, chunking, cosine similarity, retrieval, prompting and evaluation. Those concepts are not original algorithms invented by me. The available source history does not identify a separate instructor-authored algorithm file or collaborator implementation in the selected source. Course slides, assignment instructions, report drafts and exported knowledge notes are excluded rather than presented as my independent engineering work. Their exclusion does not grant reuse rights to them.

I used Codex for this preparation: source selection, dependency setup, test seams and fixtures, regression tests, the neighboring-context citation correction, safer downloader invocation, CLI setup options, comment review and documentation. This statement describes this pass; it does not establish how every earlier document or code revision was produced.

The included Harbor and Willow notes are fictional fixtures written during this preparation. They are unrelated to an employer, client or production system. The historical Wikipedia text and model weights are not distributed in this repository. The saved evaluation questions and corpus hashes allow the original development experiment to be identified without bundling that text.

Sentence Transformers, Transformers and PyTorch supply the embedding model/runtime. NumPy supplies numerical array operations. The optional downloader uses Hugging Face Datasets and Wikimedia's Wikipedia dataset. I did not train MiniLM or implement those libraries. See [data and model notes](DATA_AND_MODEL.md) for their source links and terms.

The prepared snapshot starts a separate local history so excluded dependencies, articles, personal metadata and historical objects are not part of it. It is a current source snapshot, not a reconstruction of the original development chronology. No project source-code license has been selected.
