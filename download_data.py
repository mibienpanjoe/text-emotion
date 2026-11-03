"""Download the GTSRB tables this project trains on.

The files are the public Hugging Face copy of the German Traffic Sign
Recognition Benchmark (tanganke/gtsrb). They stay in data/hf/, which git
ignores, so the repository itself stays small.
"""

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "hf"
BASE = "https://huggingface.co/datasets/tanganke/gtsrb/resolve/main/data"
FILES = {
    "train.parquet": f"{BASE}/train-00000-of-00001.parquet",
    "test.parquet": f"{BASE}/test-00000-of-00001.parquet",
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = DEST / name
        if path.is_file() and path.stat().st_size > 1_000_000:
            print(f"already present: {path}")
            continue
        print(f"downloading {name}")
        urllib.request.urlretrieve(url, path)
        print(f"saved {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
