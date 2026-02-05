import json
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.model.classic_embedding import ClassicEmbedding
from src.model.lemma_decoder import LemmaDecoder
from src.model.morph_decoder import MorphDecoder
from src.model.quantum_embedding import QuantumEmbedding
from src.model.transformer_core import TransformerCore
from src.utils import build_vocab_sizes, load_config


@dataclass
class ModelBundle:
    embedding: nn.Module
    transformer: TransformerCore
    lemma_decoder: LemmaDecoder
    morph_decoder: MorphDecoder

    def to(self, device: torch.device) -> "ModelBundle":
        self.embedding = self.embedding.to(device)
        self.transformer = self.transformer.to(device)
        self.lemma_decoder = self.lemma_decoder.to(device)
        self.morph_decoder = self.morph_decoder.to(device)
        return self

    def train(self):
        self.embedding.train()
        self.transformer.train()
        self.lemma_decoder.train()
        self.morph_decoder.train()

    def eval(self):
        self.embedding.eval()
        self.transformer.eval()
        self.lemma_decoder.eval()
        self.morph_decoder.eval()

    def parameters(self):
        return (
            list(self.embedding.parameters())
            + list(self.transformer.parameters())
            + list(self.lemma_decoder.parameters())
            + list(self.morph_decoder.parameters())
        )


def resolve_device(config: Dict) -> torch.device:
    use_cuda = config["training"]["device"] == "cuda" and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def load_vocab_size_from_config(config: Dict) -> Tuple[int, int]:
    return build_vocab_sizes(config["vocab"]["lemma_vocab"], config["vocab"]["morph_vocab"])


def load_lemma_vocab(config: Dict) -> Dict[str, int]:
    with open(config["vocab"]["lemma_vocab"], "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(config: Dict, model_type: str = "qmorph") -> ModelBundle:
    vocab_size, morph_vocab_size = load_vocab_size_from_config(config)

    emb_cfg = config["model"]["quantum_embedding"]
    if model_type == "qmorph":
        embedding = QuantumEmbedding(
            vocab_size=vocab_size,
            embed_dim=emb_cfg["embed_dim"],
            n_senses=emb_cfg["n_senses"],
            pad_idx=emb_cfg["pad_idx"],
        )
    elif model_type == "baseline":
        embedding = ClassicEmbedding(
            vocab_size=vocab_size,
            embed_dim=emb_cfg["embed_dim"],
            pad_idx=emb_cfg["pad_idx"],
        )
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    trf_cfg = config["model"]["transformer"]
    transformer = TransformerCore(
        embed_dim=trf_cfg["embed_dim"],
        num_heads=trf_cfg["num_heads"],
        ff_dim=trf_cfg["ff_dim"],
        num_layers=trf_cfg["num_layers"],
        dropout=trf_cfg["dropout"],
        max_seq_len=trf_cfg["max_seq_len"],
        is_autoregressive=True,
    )

    ld_cfg = config["model"]["lemma_decoder"]
    lemma_decoder = LemmaDecoder(
        embed_dim=ld_cfg["embed_dim"],
        vocab_size=vocab_size,
        pad_idx=ld_cfg["pad_idx"],
        dropout=ld_cfg["dropout"],
    )

    dec_cfg = config["model"]["morph_decoder"]
    morph_decoder = MorphDecoder(
        embed_dim=dec_cfg["embed_dim"],
        morph_vocab_size=morph_vocab_size,
        form_mapping_file=dec_cfg["form_mapping"],
        dropout=dec_cfg["dropout"],
        pad_idx=dec_cfg["pad_idx"],
    )

    return ModelBundle(
        embedding=embedding,
        transformer=transformer,
        lemma_decoder=lemma_decoder,
        morph_decoder=morph_decoder,
    )


def load_bundle_checkpoint(bundle: ModelBundle, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bundle.embedding.load_state_dict(checkpoint["embedding_state"])
    bundle.transformer.load_state_dict(checkpoint["transformer_state"])
    bundle.lemma_decoder.load_state_dict(checkpoint["lemma_decoder_state"])
    bundle.morph_decoder.load_state_dict(checkpoint["morph_decoder_state"])
    return checkpoint


def load_runtime_config(path: str):
    return load_config(path)
