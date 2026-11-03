"""Load GTSRB and turn sign photos into tensors the network can train on.

The usual torchvision download talks to sid.erda.dk. That host did not accept
a connection from this machine, so the images come from the public Hugging
Face copy of the same benchmark (tanganke/gtsrb): image bytes plus a class id.
"""

import io
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "hf"
IMAGE_SIZE = 64
BATCH_SIZE = 64
VAL_FRACTION = 0.2
SEED = 42

# MobileNetV2 was pretrained on ImageNet with these channel means and stds.
# Matching them keeps the pretrained features in the range they expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_TABLES = {}


def train_transform():
    """Augment training photos so the model sees more lighting and tilt.

    Signs are not mirrored: a flipped "keep right" is a different sign,
    so there is no horizontal flip here.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform():
    """Resize and normalize only. Evaluation must not randomly change the photo."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _table(split: str):
    if split not in _TABLES:
        path = DATA_DIR / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run: python3 download_data.py"
            )
        _TABLES[split] = pq.read_table(path)
    return _TABLES[split]


class SignDataset(Dataset):
    """One split of GTSRB. Each item is a PIL image (then transformed) and an int label."""

    def __init__(self, split: str, transform, indices=None):
        table = _table(split)
        self._bytes = table.column("image")
        self.labels = table.column("label").to_pylist()
        self.transform = transform
        self.indices = list(range(len(self.labels))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        raw = self._bytes[index].as_py()["bytes"]
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _split_indices(n: int):
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(n, generator=generator).tolist()
    n_val = int(n * VAL_FRACTION)
    return permutation[n_val:], permutation[:n_val]


def load_splits():
    """Return train, validation, and test datasets.

    Validation is 20% of the training images, held out so we can notice
    overfitting before we touch the test set.
    """
    n_train = _table("train").num_rows
    train_idx, val_idx = _split_indices(n_train)
    train_set = SignDataset("train", train_transform(), train_idx)
    val_set = SignDataset("train", eval_transform(), val_idx)
    test_set = SignDataset("test", eval_transform())
    return train_set, val_set, test_set


def make_loader(dataset, shuffle: bool):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def preview_dataset():
    """Training images before resize and normalization, for plotting."""
    return SignDataset("train", transform=None)
