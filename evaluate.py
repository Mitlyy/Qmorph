import argparse
import math

import torch
import torch.nn as nn

from src.data_loader import get_dataloader
from src.runtime import build_model, load_bundle_checkpoint, load_runtime_config, resolve_device
from src.utils import find_latest_checkpoint, set_seed


def evaluate_bundle(bundle, dataloader, lemma_criterion, morph_criterion, pad_idx, device):
    bundle.eval()

    total_lemma_loss = 0.0
    total_morph_loss = 0.0
    total_lemma_tokens = 0
    total_morph_tokens = 0
    correct_lemmas = 0
    correct_morphs = 0

    with torch.no_grad():
        for lemma_ids, morph_ids, _ in dataloader:
            lemma_ids = lemma_ids.to(device)
            morph_ids = morph_ids.to(device)
            padding_mask = lemma_ids.eq(pad_idx)

            psi, _ = bundle.embedding(lemma_ids)
            contextual = bundle.transformer(psi, padding_mask)

            lemma_logits = bundle.lemma_decoder(contextual)
            _, _, lemma_vocab = lemma_logits.shape
            pred_logits = lemma_logits[:, :-1, :].contiguous().view(-1, lemma_vocab)
            lemma_targets = lemma_ids[:, 1:].contiguous().view(-1)

            lemma_loss = lemma_criterion(pred_logits, lemma_targets)
            lemma_mask = lemma_targets != pad_idx
            lemma_tokens = lemma_mask.sum().item()
            total_lemma_loss += lemma_loss.item() * lemma_tokens
            total_lemma_tokens += lemma_tokens

            lemma_preds = pred_logits.argmax(dim=-1)
            correct_lemmas += (
                (lemma_preds == lemma_targets).masked_select(lemma_mask).sum().item()
            )

            morph_logits = bundle.morph_decoder(contextual)
            _, _, morph_vocab = morph_logits.shape
            morph_logits_flat = morph_logits.view(-1, morph_vocab)
            morph_targets = morph_ids.view(-1)

            morph_loss = morph_criterion(morph_logits_flat, morph_targets)
            morph_mask = morph_targets != pad_idx
            morph_tokens = morph_mask.sum().item()
            total_morph_loss += morph_loss.item() * morph_tokens
            total_morph_tokens += morph_tokens

            morph_preds = morph_logits_flat.argmax(dim=-1)
            correct_morphs += (
                (morph_preds == morph_targets).masked_select(morph_mask).sum().item()
            )

    avg_lemma_loss = total_lemma_loss / max(total_lemma_tokens, 1)
    avg_morph_loss = total_morph_loss / max(total_morph_tokens, 1)
    return {
        "lemma_loss": avg_lemma_loss,
        "perplexity": math.exp(avg_lemma_loss),
        "lemma_accuracy": correct_lemmas / max(total_lemma_tokens, 1) * 100,
        "morph_loss": avg_morph_loss,
        "morph_accuracy": correct_morphs / max(total_morph_tokens, 1) * 100,
    }


def print_metrics(metrics, model_type):
    print(f"model={model_type}")
    print(f"  next-lemma loss      : {metrics['lemma_loss']:.4f}")
    print(f"  next-lemma perplexity: {metrics['perplexity']:.2f}")
    print(f"  next-lemma accuracy  : {metrics['lemma_accuracy']:.2f}%")
    print(f"  morph loss           : {metrics['morph_loss']:.4f}")
    print(f"  morph accuracy       : {metrics['morph_accuracy']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="evaluate model")
    parser.add_argument("--config", type=str, default="config/config_books.yaml")
    parser.add_argument("--model-type", type=str, default="qmorph", choices=["qmorph", "baseline"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    set_seed(config.get("seed", 42))
    device = resolve_device(config)

    test_loader = get_dataloader(
        data_file=config["data"]["test_file"],
        lemma_vocab_file=config["vocab"]["lemma_vocab"],
        morph_vocab_file=config["vocab"]["morph_vocab"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        max_length=config["training"]["max_length"],
        num_workers=0,
    )

    bundle = build_model(config, model_type=args.model_type).to(device)

    checkpoint = args.checkpoint or find_latest_checkpoint(
        config["checkpoint"]["dir"], f"{args.model_type}_model"
    )
    load_bundle_checkpoint(bundle, checkpoint, device)

    pad_idx = config["model"]["morph_decoder"]["pad_idx"]
    lemma_criterion = nn.CrossEntropyLoss(ignore_index=config["model"]["lemma_decoder"]["pad_idx"])
    morph_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    metrics = evaluate_bundle(
        bundle, test_loader, lemma_criterion, morph_criterion, pad_idx, device
    )
    print_metrics(metrics, args.model_type)


if __name__ == "__main__":
    main()
