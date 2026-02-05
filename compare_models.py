import argparse
import json

import torch.nn as nn

from evaluate import evaluate_bundle
from src.data_loader import get_dataloader
from src.runtime import build_model, load_bundle_checkpoint, load_runtime_config, resolve_device
from src.utils import find_latest_checkpoint, set_seed


def main():
    parser = argparse.ArgumentParser(description="compare qmorph against baseline")
    parser.add_argument("--config", type=str, default="config/config_books.yaml")
    parser.add_argument("--qmorph-checkpoint", type=str, default=None)
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/model_comparison.json")
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    set_seed(config.get("seed", 42))
    device = resolve_device(config)

    loader = get_dataloader(
        data_file=config["data"]["test_file"],
        lemma_vocab_file=config["vocab"]["lemma_vocab"],
        morph_vocab_file=config["vocab"]["morph_vocab"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        max_length=config["training"]["max_length"],
        num_workers=0,
    )

    pad_idx = config["model"]["morph_decoder"]["pad_idx"]
    lemma_criterion = nn.CrossEntropyLoss(ignore_index=config["model"]["lemma_decoder"]["pad_idx"])
    morph_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    qmorph = build_model(config, "qmorph").to(device)
    baseline = build_model(config, "baseline").to(device)

    qmorph_ckpt = args.qmorph_checkpoint or find_latest_checkpoint(config["checkpoint"]["dir"], "qmorph_model")
    baseline_ckpt = args.baseline_checkpoint or find_latest_checkpoint(config["checkpoint"]["dir"], "baseline_model")

    load_bundle_checkpoint(qmorph, qmorph_ckpt, device)
    load_bundle_checkpoint(baseline, baseline_ckpt, device)

    qmorph_metrics = evaluate_bundle(qmorph, loader, lemma_criterion, morph_criterion, pad_idx, device)
    baseline_metrics = evaluate_bundle(baseline, loader, lemma_criterion, morph_criterion, pad_idx, device)

    comparison = {
        "qmorph": qmorph_metrics,
        "baseline": baseline_metrics,
        "improvement": {
            "perplexity_delta": baseline_metrics["perplexity"] - qmorph_metrics["perplexity"],
            "lemma_accuracy_delta": qmorph_metrics["lemma_accuracy"] - baseline_metrics["lemma_accuracy"],
            "morph_accuracy_delta": qmorph_metrics["morph_accuracy"] - baseline_metrics["morph_accuracy"],
        },
        "winner": "qmorph" if qmorph_metrics["perplexity"] <= baseline_metrics["perplexity"] else "baseline",
    }

    print(json.dumps(comparison, ensure_ascii=False, indent=2))

    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
