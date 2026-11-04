"""Classify one message with the trained model."""

import argparse
from pathlib import Path

import torch

from data import Vocabulary
from emotions import EMOTIONS
from model import EmotionLSTM

CHECKPOINT = Path(__file__).resolve().parent / "outputs" / "model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    vocabulary = Vocabulary(saved["vocab"])
    network = EmotionLSTM(len(vocabulary), pad_id=vocabulary.pad_id).to(device)
    network.load_state_dict(saved["state_dict"])
    network.eval()

    ids = vocabulary.encode(args.text)
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long)

    with torch.no_grad():
        probabilities = torch.softmax(network(token_ids, lengths), dim=1)[0]

    top = torch.topk(probabilities, k=3)
    print(f"text: {args.text}")
    for score, index in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {score:6.1%}  {EMOTIONS[index]}")


if __name__ == "__main__":
    main()
