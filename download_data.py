"""Download the emotion tables this project trains on.

The files are the public Hugging Face copy of the Emotion dataset
(dair-ai/emotion), split configuration: 16,000 train, 2,000 validation,
2,000 test. They stay in data/hf/, which git ignores.
"""

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "hf"
BASE = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split"
FILES = {
    "train.parquet": f"{BASE}/train-00000-of-00001.parquet",
    "validation.parquet": f"{BASE}/validation-00000-of-00001.parquet",
    "test.parquet": f"{BASE}/test-00000-of-00001.parquet",
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = DEST / name
        if path.is_file() and path.stat().st_size > 10_000:
            print(f"already present: {path}")
            continue
        print(f"downloading {name}")
        urllib.request.urlretrieve(url, path)
        print(f"saved {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
