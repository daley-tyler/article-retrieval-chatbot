"""Check corpus identity without copying or modifying article text."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "docs/corpus-manifest.json").read_text())
    expected = manifest["files"]
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.documents.glob("*.txt")}
    if actual != {name: row["sha256"] for name, row in expected.items()}:
        raise SystemExit("Corpus differs from the historical snapshot; do not reuse its recorded results.")
    print(f"All {len(actual)} historical corpus hashes match.")


if __name__ == "__main__":
    main()
