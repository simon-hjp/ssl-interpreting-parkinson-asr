"""
Finetune selected XLS-R encoder layers on the train split,
with validation and early stopping based on the validation split.

Example usage
-------
python scripts/runs/experiments/finetuned_cross_full/tune_backbone.py \
       --splits-dir splits/gita \
       --audio-dir data/gita/norm_audios \
       --unfreeze-layers 1 2 3 4 \
       --layer-n 2 \
       --max-epochs 20 \
       --patience 3 \
       --lr 1e-6 \
       --device cuda

"""

import argparse
import csv
import pathlib
import time
import random
import torch
from torch import nn
import torchaudio  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from transformers import Wav2Vec2Model


def set_seed(seed: int = 42):
    """Set seed everywhere for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wave(path: pathlib.Path) -> torch.Tensor:
    """Load and verify .wav file"""
    wav, sr = torchaudio.load(path)
    if sr != 16_000:
        raise ValueError(f"Audio wav at {path} not normalized, sr = {sr}")
    return wav.squeeze(0)


def tune_fold(
    fold_idx: int,
    audio_dir,
    splits_dir,
    unfreeze_layers,
    layer_n,
    lr,
    max_epochs,
    patience,
    device,
) -> None:
    # setup and model loading
    fold_dir = pathlib.Path(splits_dir) / f"fold_{fold_idx}"
    train_csv = fold_dir / "train.csv"
    valid_csv = fold_dir / "validation.csv"

    with open(train_csv, encoding="utf8") as train_f:
        train_rows = list(csv.DictReader(train_f))
    with open(valid_csv, encoding="utf8") as val_f:
        valid_rows = list(csv.DictReader(val_f))

    device = torch.device(device)

    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", output_hidden_states=True
    ).to(device)  # type: ignore

    # enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # freeze model, then unfreeze layers to be tuned
    for p in model.parameters():
        p.requires_grad = False

    for i in unfreeze_layers:
        for p in model.encoder.layers[i].parameters():
            p.requires_grad = True

    # classifier head
    hidden_size = model.config.hidden_size
    cls_head = nn.Linear(hidden_size, 2, bias=False).to(device)

    # optimizer
    opt = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(cls_head.parameters()),
        lr=lr,
        weight_decay=1e-2,
    )
    loss_fn = nn.CrossEntropyLoss()
    label_map = {"0": 0, "1": 1}

    best_val_f1 = -1.0
    no_improve = 0

    # keep track of metrics
    metrics_file = pathlib.Path("checkpoints") / f"xlsr_metrics_fold{fold_idx}.csv"
    epoch_metrics = []

    print(
        f"Fold {fold_idx}: finetuning on {len(train_rows)} train / {len(valid_rows)} val"
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        cls_head.train()
        train_loss = 0.0
        start = time.time()

        # training loop
        for r in tqdm(
            train_rows,
            desc=f"Train fold {fold_idx} epoch {epoch}/{max_epochs}",
            leave=False,
        ):
            wav_path = pathlib.Path(audio_dir) / f"{r['sample_id']}.wav"
            wav = load_wave(wav_path).to(device)
            target = torch.tensor([label_map[r["label"]]], device=device)

            out = model(wav.unsqueeze(0), output_hidden_states=True)
            rep = out.hidden_states[layer_n].mean(dim=1)  # (1, D)
            logits = cls_head(rep)
            loss = loss_fn(logits, target)

            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()

        mean_train_loss = train_loss / len(train_rows)
        dur = time.time() - start

        # validation loop
        model.eval()
        cls_head.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for r in tqdm(
                valid_rows,
                desc=f"Val fold{fold_idx} ep{epoch}/{max_epochs}",
                leave=False,
            ):
                wav_path = pathlib.Path(audio_dir) / f"{r['sample_id']}.wav"
                wav = load_wave(wav_path).to(device)
                target = torch.tensor([label_map[r["label"]]], device=device)

                out = model(wav.unsqueeze(0), output_hidden_states=True)
                rep = out.hidden_states[layer_n].mean(dim=1)
                logits = cls_head(rep)
                loss = loss_fn(logits, target)

                val_loss += loss.item()
                y_true.append(target.item())
                y_pred.append(logits.argmax(dim=-1).item())

        mean_val_loss = val_loss / len(valid_rows)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        epoch_metrics.append((epoch, mean_val_loss, val_f1))

        print(
            f"epoch {epoch}/{max_epochs} | "
            f"train_loss {mean_train_loss:.4f} | val_loss {mean_val_loss:.4f} | {dur:.1f}s | "
            f"val_F1 {val_f1:.3f} | time {dur:.1f}s"
        )

        # early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0

            # save checkpoint for this fold
            ckpt_dir = pathlib.Path("checkpoints") / f"xlsr_fold{fold_idx}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)

            # save metrics
            with open(metrics_file, "w", encoding="utf8") as f:
                f.write("epoch,val_loss,val_f1\n")
                f.write(f"{epoch},{mean_val_loss:.4f},{val_f1:.4f}\n")

            print(f"Improved val_f1, saved checkpoint: {ckpt_dir}")

        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping.")
                break

    # fallback checkpoint saving
    if best_val_f1 == -1.0:
        ckpt_dir = pathlib.Path("checkpoints") / f"xlsr_fold{fold_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        with open(metrics_file, "w", encoding="utf8") as f:
            f.write("epoch,val_loss,val_f1\n")
            f.write(f"NA,{mean_val_loss:.4f},{val_f1:.4f}\n")
        print(
            f"    No validation improvement seen, saved final checkpoint and metrics: {ckpt_dir}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        nargs="+",
        help="Encoder layers to unfreeze (e.g., 1 2 3)",
    )
    parser.add_argument(
        "--layer-n",
        type=int,
        default=4,
        help="Layer index to extract features from",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=20, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Epochs to wait without improvement before stopping",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    for fold in range(5):
        tune_fold(
            fold,
            args.audio_dir,
            args.splits_dir,
            args.unfreeze_layers,
            args.layer_n,
            args.lr,
            args.max_epochs,
            args.patience,
            args.device,
        )
