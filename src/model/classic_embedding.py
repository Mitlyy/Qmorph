import torch
import torch.nn as nn


class ClassicEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, token_ids: torch.LongTensor):
        vectors = self.embedding(token_ids)
        return vectors, None
