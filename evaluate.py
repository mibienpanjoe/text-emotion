"""Score the saved model on the official test split."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data import EmotionDataset, Vocabulary, make_loader
from emotions import EMOTIONS
from model import EmotionLSTM

OUTPUT = Path(__file__).resolve().parent / "outputs"
CHECKPOINT = OUTPUT / "model.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    vocabulary = Vocabulary(saved["vocab"])
    test_set = EmotionDataset("test", vocabulary)
    loader = make_loader(test_set, shuffle=False)

    network = EmotionLSTM(len(vocabulary), pad_id=vocabulary.pad_id).to(device)
    network.load_state_dict(saved["state_dict"])
    network.eval()

    truths = []
    predictions = []
    with torch.no_grad():
        for token_ids, lengths, labels in loader:
            logits = network(token_ids.to(device), lengths)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            truths.extend(labels.tolist())

    accuracy = np.mean(np.array(predictions) == np.array(truths))
    report = classification_report(
        truths, predictions, target_names=EMOTIONS, digits=3, zero_division=0
    )
    print(f"test texts: {len(test_set)}")
    print(f"test accuracy: {accuracy:.3f}")
    print(report)

    OUTPUT.mkdir(exist_ok=True)
    report_path = OUTPUT / "classification_report.txt"
    report_path.write_text(f"test accuracy: {accuracy:.3f}\n\n{report}")

    matrix = confusion_matrix(truths, predictions)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"Test confusion matrix  (accuracy {accuracy:.1%})")
    axis.set_xlabel("predicted emotion")
    axis.set_ylabel("true emotion")
    axis.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    axis.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure_path = OUTPUT / "confusion_matrix.png"
    figure.savefig(figure_path, dpi=120)
    print(f"saved {report_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
