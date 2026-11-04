"""An embedding and an LSTM that score six emotions."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from emotions import EMOTIONS


class EmotionLSTM(nn.Module):
    """Read a sentence word by word and classify the final memory.

    Padding positions are packed away, so they do not change that memory.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, len(EMOTIONS))

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(hidden[-1])
