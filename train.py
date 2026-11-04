"""Train the emotion classifier and keep the best validation accuracy."""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from data import load_splits, make_loader
from emotions import EMOTIONS
from model import EmotionLSTM

OUTPUT = Path(__file__).resolve().parent / "outputs"
CHECKPOINT = OUTPUT / "model.pt"


def class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    """Give rarer emotions a larger share of the loss."""
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (len(EMOTIONS) * counts[index]) for index in range(len(EMOTIONS))]
    return torch.tensor(weights, dtype=torch.float, device=device)


def run_epoch(network, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    network.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0

    for token_ids, lengths, labels in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = network(token_ids, lengths)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_set, val_set, _, vocabulary = load_splits()
    print(f"train texts: {len(train_set)}   validation texts: {len(val_set)}")
    print(f"vocabulary size: {len(vocabulary)}")

    network = EmotionLSTM(len(vocabulary), pad_id=vocabulary.pad_id).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set.labels, device))
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    OUTPUT.mkdir(exist_ok=True)
    best_accuracy = 0.0
    train_loader = make_loader(train_set, shuffle=True)
    val_loader = make_loader(val_set, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(network, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(network, val_loader, criterion, device)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train loss {train_loss:.3f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}"
        )
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(
                {
                    "state_dict": network.state_dict(),
                    "vocab": vocabulary.tokens,
                    "val_accuracy": val_acc,
                },
                CHECKPOINT,
            )
            print(f"  saved {CHECKPOINT}")

    print(f"best validation accuracy: {best_accuracy:.3f}")


if __name__ == "__main__":
    main()
