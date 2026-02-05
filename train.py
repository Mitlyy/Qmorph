import argparse
import glob
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.data_loader import get_dataloader
from src.runtime import build_model, load_bundle_checkpoint, load_runtime_config, resolve_device
from src.utils import set_seed


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    return LambdaLR(
        optimizer,
        lambda step: (
            step / warmup_steps
            if step < warmup_steps
            else max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))
        ),
    )


def save_checkpoint(path, epoch, global_step, bundle, optimizer, scheduler):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "embedding_state": bundle.embedding.state_dict(),
            "transformer_state": bundle.transformer.state_dict(),
            "lemma_decoder_state": bundle.lemma_decoder.state_dict(),
            "morph_decoder_state": bundle.morph_decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )


def train_one_epoch(bundle, train_loader, optimizer, scheduler, lemma_criterion, morph_criterion, pad_idx, grad_clip, log_interval, device, global_step):
    bundle.train()
    total_lemma_loss = 0.0
    total_morph_loss = 0.0

    for batch_idx, (lemma_ids, morph_ids, _) in enumerate(train_loader, 1):
        lemma_ids = lemma_ids.to(device)
        morph_ids = morph_ids.to(device)

        optimizer.zero_grad()
        psi, _ = bundle.embedding(lemma_ids)
        contextual = bundle.transformer(psi, lemma_ids.eq(pad_idx))

        lemma_logits = bundle.lemma_decoder(contextual)
        bsz, seq_len, vocab_size = lemma_logits.shape
        pred_logits = lemma_logits[:, :-1, :].contiguous().view(-1, vocab_size)
        lemma_targets = lemma_ids[:, 1:].contiguous().view(-1)
        lemma_loss = lemma_criterion(pred_logits, lemma_targets)

        morph_logits = bundle.morph_decoder(contextual)
        _, _, morph_vocab_size = morph_logits.shape
        morph_loss = morph_criterion(
            morph_logits.view(-1, morph_vocab_size),
            morph_ids.view(-1),
        )

        loss = lemma_loss + morph_loss
        loss.backward()
        nn.utils.clip_grad_norm_(bundle.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        total_lemma_loss += lemma_loss.item()
        total_morph_loss += morph_loss.item()

        if batch_idx % log_interval == 0:
            print(
                f"step={global_step} batch={batch_idx} lemma_loss={lemma_loss.item():.4f} morph_loss={morph_loss.item():.4f}"
            )

    return global_step, total_lemma_loss / max(len(train_loader), 1), total_morph_loss / max(len(train_loader), 1)


def main():
    parser = argparse.ArgumentParser(description="train qmorph or baseline model")
    parser.add_argument("--config", type=str, default="config/config_books.yaml")
    parser.add_argument("--model-type", type=str, default="qmorph", choices=["qmorph", "baseline"])
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    set_seed(config.get("seed", 42))
    device = resolve_device(config)

    train_loader = get_dataloader(
        data_file=config["data"]["train_file"],
        lemma_vocab_file=config["vocab"]["lemma_vocab"],
        morph_vocab_file=config["vocab"]["morph_vocab"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        max_length=config["training"]["max_length"],
        num_workers=0,
    )

    bundle = build_model(config, model_type=args.model_type).to(device)

    opt_params = config["optimizer"]["params"].copy()
    opt_params["lr"] = float(opt_params["lr"])
    opt_params["eps"] = float(opt_params["eps"])
    opt_params["weight_decay"] = float(opt_params["weight_decay"])
    opt_params["betas"] = tuple(opt_params["betas"])
    optimizer = AdamW(bundle.parameters(), **opt_params)

    scheduler = make_scheduler(
        optimizer,
        warmup_steps=config["scheduler"]["params"]["warmup_steps"],
        total_steps=config["scheduler"]["params"]["total_steps"],
    )

    pad_idx = config["model"]["morph_decoder"]["pad_idx"]
    lemma_criterion = nn.CrossEntropyLoss(ignore_index=config["model"]["lemma_decoder"]["pad_idx"])
    morph_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    ckpt_dir = config["checkpoint"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    prefix = f"{args.model_type}_model"
    keep_last = config["checkpoint"].get("keep_last", 5)

    global_step = 0
    start_epoch = 1

    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch_*.pt")))
    if checkpoints:
        state = load_bundle_checkpoint(bundle, checkpoints[-1], device)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        print(f"resume from {checkpoints[-1]}")

    for epoch in range(start_epoch, config["training"]["num_epochs"] + 1):
        global_step, avg_lemma_loss, avg_morph_loss = train_one_epoch(
            bundle,
            train_loader,
            optimizer,
            scheduler,
            lemma_criterion,
            morph_criterion,
            pad_idx,
            config["training"]["grad_clip"],
            config["training"]["log_interval"],
            device,
            global_step,
        )
        ckpt_path = os.path.join(ckpt_dir, f"{prefix}_epoch_{epoch}.pt")
        save_checkpoint(ckpt_path, epoch, global_step, bundle, optimizer, scheduler)
        print(f"epoch={epoch} avg_lemma_loss={avg_lemma_loss:.4f} avg_morph_loss={avg_morph_loss:.4f}")

        old = sorted(glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch_*.pt")))
        if len(old) > keep_last:
            for stale in old[:-keep_last]:
                os.remove(stale)


if __name__ == "__main__":
    main()
