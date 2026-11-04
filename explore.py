"""Look at the dataset before any training: counts and a few messages."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from data import load_texts
from emotions import EMOTIONS

OUTPUT = Path(__file__).resolve().parent / "outputs"


def main():
    texts, labels = load_texts("train")
    counts = Counter(labels)
    print(f"training texts: {len(texts)}")
    print(f"emotions: {len(EMOTIONS)}")
    print()
    for class_id, name in enumerate(EMOTIONS):
        print(f"{class_id}  {counts[class_id]:5d}  {name}")
        shown = 0
        for text, label in zip(texts, labels):
            if label != class_id:
                continue
            print(f"     {text}")
            shown += 1
            if shown == 2:
                break
        print()

    OUTPUT.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    names = EMOTIONS
    values = [counts[index] for index in range(len(EMOTIONS))]
    axis.bar(names, values, color="#3d5a80")
    axis.set_title("Training texts per emotion")
    axis.set_ylabel("texts")
    figure.tight_layout()
    figure_path = OUTPUT / "class_counts.png"
    figure.savefig(figure_path, dpi=120)
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
