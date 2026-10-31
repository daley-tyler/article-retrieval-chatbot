# Data and model notes

The included Harbor and Willow files are fictional examples written for this project. They contain no company records or production configuration.

The historical article experiment used text downloaded from [Wikimedia's Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia), configuration `20231101.simple`. Its dataset card lists CC BY-SA 3.0 and GFDL terms. The 37 local article texts are excluded from this repository. Filenames, sizes and SHA-256 hashes are retained solely to identify the experiment. If downloading, adapting or redistributing Wikipedia text, follow the upstream attribution and license requirements; this repository does not grant additional rights to it.

The optional downloader is invoked explicitly and never runs on import or during unit tests:

```sh
python -m pip install -r requirements-data.txt
python download_articles.py --output article_txt
```

It selects the first 40 rows at dataset revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` and refuses to overwrite an existing output directory. This revision was identified from the original local cache. Exact equivalence to the historical corpus has not been verified. The optional data dependencies have not received the same clean-environment validation as the core runtime.

The encoder is [Sentence Transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), whose model card labels it Apache 2.0. The prepared default pins revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, the revision available in the original local cache. Model weights are downloaded separately and are not part of this repository. The project uses the pretrained model for embeddings; no model training or fine-tuning is implemented.

The default runtime may contact Hugging Face to obtain public model files. `--local-files-only`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` require cached files and avoid model downloads. No LLM API, hosted vector store or paid API credential is needed. No access tokens should be added to source files, test fixtures or commits.

Third-party libraries remain under their own terms: [Sentence Transformers](https://github.com/huggingface/sentence-transformers), [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch), [NumPy](https://github.com/numpy/numpy) and, for the optional downloader, [Datasets](https://github.com/huggingface/datasets). They are installed as dependencies, not vendored into the source tree. No source-code reuse license has been assigned to this project.
