"""
24-layer probing experiment of Wav2Vec2.0 XLS-R on PC-GITA,
with per-task and pooled evaluation
Outputs:
- results/probe_layers_<TASK>.png
- results/probe_layers_all.csv
"""

import argparse
import pathlib
import csv
import numpy as np
import torch
import torchaudio  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore


def load_wave(path):
    """
    Load and verify .wav file
    """
    wav, sr = torchaudio.load(path)
    if sr != 16_000:
        print(f"Warning: wav at {path} does not seem to be normalized")
    return wav.squeeze(0)


def get_hidden_states(model, wave_1d, device):
    """
    Returns a list of 24 transformer layer tensors on CPU
    """
    wav = wave_1d.unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.extract_features(wav)[0]  # get feature vectors
    if isinstance(feats, list) and len(feats) == 24:
        return [f.squeeze(0).cpu() for f in feats]
    raise RuntimeError("torchaudio did not return all encoder layers")


def probe_once(wavs, labels, device, model, seeds, tag):
    """Return mean and std F1 for all layers, averaged over seeds and folds"""
    all_f1 = np.zeros((len(seeds), 24))
    for seed_idx, seed in enumerate(seeds):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        f1_layers = np.zeros(24)

        for fold_id, (tr, te) in enumerate(cv.split(wavs, labels)):
            print(f"{tag} | seed {seed} | fold {fold_id+1}/5")
            X_train, y_train = [wavs[i] for i in tr], labels[tr]
            X_test, y_test = [wavs[i] for i in te], labels[te]

            h_train, h_test = [], []
            with torch.no_grad():
                # get training embeddings
                for i, p in enumerate(X_train, 1):
                    if i % 5 == 0 or i == len(X_train):
                        print(f"    - Train {i}/{len(X_train)}", end="\r")
                    hidden_states = get_hidden_states(model, load_wave(p), device)
                    h_train.append([o.squeeze(0).mean(0).cpu() for o in hidden_states])
                print()
                # get testing embeddings
                for i, p in enumerate(X_test, 1):
                    if i % 5 == 0 or i == len(X_test):
                        print(f"    - Test  {i}/{len(X_test)}", end="\r")
                    hidden_states = get_hidden_states(model, load_wave(p), device)
                    h_test.append([o.squeeze(0).mean(0).cpu() for o in hidden_states])
                print()

            # convert to numpy for sklearn compatibility
            h_train, h_test = np.stack(h_train), np.stack(h_test)

            # classify each of the 24 layer embeddings
            for L in range(24):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                clf.fit(h_train[:, L, :], y_train)
                f1_layers[L] += f1_score(
                    y_test, clf.predict(h_test[:, L, :]), average="macro"
                )

        all_f1[seed_idx] = f1_layers / 5  # average over folds

    mean = all_f1.mean(0)
    std = all_f1.std(0)

    # print results
    print("\n Layer | MeanF1 (Std)")
    for L, (m, s) in enumerate(zip(mean, std)):
        print(f" {L:02d} | {m:5.3f} ({s:4.3f})")
    print(" -" * 10)
    return mean, std


def perform_probe(audio_files, splits):
    """
    Use a cross-validation to fit a logistic regression model
    """
    seeds = [12, 21, 33, 42, 52]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
    model = bundle.get_model().to(device).eval()

    with open(splits, encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    tasks = sorted({r["task_id"] for r in rows})
    tasks.append("ALL")  # pooled set

    wavs = [pathlib.Path(audio_files) / f"{r['sample_id']}.wav" for r in rows]
    labels = np.array([int(r["label"]) for r in rows])

    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    layer_axis = np.arange(24)

    # store per-task arrays to write one CSV later
    csv_columns = ["layer"]
    csv_matrix = [layer_axis.tolist()]

    for task in tasks:
        idxs = (
            range(len(rows))
            if task == "ALL"
            else [i for i, r in enumerate(rows) if r["task_id"] == task]
        )

        print(f"\nProbing task: {task} ({len(idxs)} samples)")
        mean, std = probe_once(
            [wavs[i] for i in idxs], labels[idxs], device, model, seeds, task
        )

        # append to CSV matrix
        csv_columns += [f"{task}_mean", f"{task}_std"]
        csv_matrix[0] += mean.tolist() + std.tolist()

    # write to csv
    out_csv = results_dir / "probe_layers_all.csv"
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(csv_columns)
        w.writerow(csv_matrix[0])
    print(f"\nCombined CSV saved: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-audios", required=True)
    ap.add_argument("--splits-file", required=True)
    parsed_args = ap.parse_args()
    perform_probe(parsed_args.norm_audios, parsed_args.splits_file)
