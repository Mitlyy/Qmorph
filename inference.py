import argparse
import os
import sys
from typing import List

import torch

sys.path.append(os.path.dirname(__file__))

from src.preprocessing.lemmatizer import Lemmatizer
from src.preprocessing.tokenizer import Tokenizer
from src.runtime import (
    build_model,
    load_bundle_checkpoint,
    load_lemma_vocab,
    load_runtime_config,
    resolve_device,
)
from src.utils import find_latest_checkpoint, set_seed


def preprocess(
    text: str,
    tokenizer: Tokenizer,
    lemmatizer: Lemmatizer,
    lemma_vocab: dict,
    max_length: int,
):
    tokens = tokenizer.tokenize(text)[:max_length]
    lemma_ids = []
    for token in tokens:
        lemma, _ = lemmatizer.lemmatize_token(token)
        lemma_ids.append(lemma_vocab.get(lemma, lemma_vocab.get("<unk>", 1)))
    return tokens, torch.tensor([lemma_ids], dtype=torch.long)


def generate_lemmas(
    start_ids: torch.LongTensor,
    embedding,
    transformer,
    lemma_decoder,
    cfg: dict,
    device: torch.device,
) -> List[int]:
    max_len = cfg["max_gen_length"]
    decoder_type = cfg.get("decoder_type", "greedy")
    if decoder_type == "beam":
        return lemma_decoder.beam_search_decode(
            embedding,
            transformer,
            start_ids.to(device),
            max_length=max_len,
            beam_size=cfg.get("beam_size", 5),
            device=device,
        )
    return lemma_decoder.greedy_decode(
        embedding,
        transformer,
        start_ids.to(device),
        max_length=max_len,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--config", type=str, default="config/config_books.yaml")
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="qmorph", choices=["qmorph", "baseline"])
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    set_seed(config.get("seed", 42))
    device = resolve_device(config)

    bundle = build_model(config, model_type=args.model_type).to(device)

    ckpt_path = args.checkpoint or find_latest_checkpoint(
        config["checkpoint"]["dir"], f"{args.model_type}_model"
    )
    load_bundle_checkpoint(bundle, ckpt_path, device)
    bundle.eval()

    tokenizer = Tokenizer()
    lemmatizer = Lemmatizer()
    lemma_vocab = load_lemma_vocab(config)

    tokens, start_ids = preprocess(
        args.sentence,
        tokenizer,
        lemmatizer,
        lemma_vocab,
        config["training"]["max_length"],
    )

    generated_ids = generate_lemmas(
        start_ids,
        bundle.embedding,
        bundle.transformer,
        bundle.lemma_decoder,
        config["inference"],
        device,
    )

    full_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        psi, _ = bundle.embedding(full_ids)
        contextual = bundle.transformer(
            psi, full_ids.eq(config["model"]["quantum_embedding"]["pad_idx"])
        )
        morph_logits = bundle.morph_decoder(contextual)
        forms = bundle.morph_decoder.decode_forms(full_ids, morph_logits)[0]

    print(f">>> Input tokens       : {tokens}")
    print(f">>> Generated lemma IDs: {generated_ids}")
    print(f">>> Output sentence    : {' '.join([word for word in forms if word])}")


if __name__ == "__main__":
    main()
