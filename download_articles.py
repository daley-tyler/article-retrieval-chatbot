"""Download an optional Wikipedia subset into a new local directory."""
import argparse
from pathlib import Path
import re

NUM_ARTICLES = 40
DATASET_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"

def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:120] if name else "article"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("article_txt"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new directory to preserve the existing corpus.")

    from datasets import load_dataset

    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple",
                           split=f"train[:{NUM_ARTICLES}]", revision=DATASET_REVISION)
    args.output.mkdir(parents=True, exist_ok=False)
    for i, row in enumerate(dataset):
        title = row.get("title", f"article_{i}")
        text = row.get("text", "").strip()
        if not text:
            continue
        filename = safe_filename(title) + ".txt"
        with (args.output / filename).open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(f"Saved {len(list(args.output.glob('*.txt')))} files to {args.output}")


if __name__ == "__main__":
    main()
