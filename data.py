"""Load the emotion texts and turn them into padded word-id sequences."""

import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "hf"

PAD = "<pad>"
UNK = "<unk>"
MIN_FREQ = 2
MAX_LEN = 40
BATCH_SIZE = 64

TOKEN_RE = re.compile(r"[a-z0-9']+")
_TABLES = {}


def tokenize(text: str) -> list[str]:
    """Lowercase the message and keep words, numbers, and apostrophes."""
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Map between words and integer ids. Index 0 is padding, 1 is unknown."""

    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    def encode(self, text: str) -> list[int]:
        unknown = self.token_to_id[UNK]
        ids = [self.token_to_id.get(token, unknown) for token in tokenize(text)]
        if not ids:
            ids = [unknown]
        return ids[:MAX_LEN]


def build_vocabulary(texts: list[str]) -> Vocabulary:
    """Count words on the training texts only. Rare words become unknown."""
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    tokens = [PAD, UNK]
    tokens.extend(token for token, count in counts.most_common() if count >= MIN_FREQ)
    return Vocabulary(tokens)


def _table(split: str):
    if split not in _TABLES:
        path = DATA_DIR / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run: python3 download_data.py")
        _TABLES[split] = pq.read_table(path)
    return _TABLES[split]


def load_texts(split: str) -> tuple[list[str], list[int]]:
    table = _table(split)
    texts = table.column("text").to_pylist()
    labels = table.column("label").to_pylist()
    return texts, labels


class EmotionDataset(Dataset):
    """One split. Each item is a 1-D LongTensor of word ids and an int label."""

    def __init__(self, split: str, vocabulary: Vocabulary):
        self.texts, self.labels = load_texts(split)
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ids = torch.tensor(self.vocabulary.encode(self.texts[index]), dtype=torch.long)
        return ids, self.labels[index]


def collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def load_splits():
    """Train, validation, and test, sharing a vocabulary built from train only."""
    train_texts, _ = load_texts("train")
    vocabulary = build_vocabulary(train_texts)
    return (
        EmotionDataset("train", vocabulary),
        EmotionDataset("validation", vocabulary),
        EmotionDataset("test", vocabulary),
        vocabulary,
    )


def make_loader(dataset, shuffle: bool):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate)
