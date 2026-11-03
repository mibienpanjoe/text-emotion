"""Classify one sign photo with the trained model."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from class_names import CLASS_NAMES
from data import eval_transform
from model import build_model

CHECKPOINT = Path(__file__).resolve().parent / "outputs" / "model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = build_model().to(device)
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    network.load_state_dict(saved["state_dict"])
    network.eval()

    image = Image.open(args.image).convert("RGB")
    batch = eval_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(network(batch), dim=1)[0]

    top = torch.topk(probabilities, k=3)
    print(f"file: {args.image}")
    for score, index in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {score:6.1%}  {CLASS_NAMES[index]}")


if __name__ == "__main__":
    main()
