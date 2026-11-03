"""Train the sign classifier and keep the weights with the best validation accuracy."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import load_splits, make_loader
from model import build_model

OUTPUT = Path(__file__).resolve().parent / "outputs"
CHECKPOINT = OUTPUT / "model.pt"


def run_epoch(network, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    network.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = network(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)

    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_set, val_set, _ = load_splits()
    train_loader = make_loader(train_set, shuffle=True)
    val_loader = make_loader(val_set, shuffle=False)
    print(f"train images: {len(train_set)}   val images: {len(val_set)}")

    network = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    OUTPUT.mkdir(exist_ok=True)
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            network, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(network, val_loader, criterion, device)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train loss {train_loss:.3f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}"
        )
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(
                {"state_dict": network.state_dict(), "val_accuracy": val_acc},
                CHECKPOINT,
            )
            print(f"  saved {CHECKPOINT}")

    print(f"best validation accuracy: {best_accuracy:.3f}")


if __name__ == "__main__":
    main()
