import os
import argparse
import glob
import math
import pathlib
import numpy as np
import torch
import torchaudio  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import Wav2Vec2Model


def load_wave(path: pathlib.Path, target_sr: int) -> torch.Tensor:
    """Load .wav file and verify its sample rate"""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        raise ValueError(f"wav file at {path} has incorrect sr: {sr}")
    return wav.squeeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WAV2VEC-based Feature Embedding Extraction from any layer"
    )
    parser.add_argument(
        "--norm-audios",
        type=str,
        required=True,
        help="Directory containing normalized .wav files"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Maximum chunk length in seconds, default: 30"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to create embedding subfolders"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned XLS-R checkpoint"
    )
    parser.add_argument(
        "--layer-n",
        type=int,
        default=2,
        help="Hidden layer to extract (0-23)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    args = parser.parse_args()
    
    checkpoint = args.checkpoint
    output_dir = args.output_dir
    norm_audios = args.norm_audios
    layer_n = args.layer_n
    chunk_size = args.chunk_size

    device = torch.device(args.device)
    print(f"Using device {device}")

    # load model
    print("Loading fine-tuned checkpoint from", checkpoint)
    model = Wav2Vec2Model.from_pretrained(
        checkpoint,
        output_hidden_states=True
    ).to(device)  # type: ignore
    model.eval()
    target_rate = 16_000

    # prepare output directory
    layer_str = str(layer_n).zfill(2)
    feature_output_dir = pathlib.Path(output_dir) / f"layer{layer_str}"
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    # get input files
    wav_paths = sorted(glob.glob(f"{norm_audios}{os.sep}*.wav"))

    # loop over each file
    for wav_path in tqdm(wav_paths, desc="Extracting"):
        wav_path = pathlib.Path(wav_path)
        sample_id = wav_path.stem + ".npz"

        waveform = load_wave(wav_path, target_rate).to(device)
        length_seconds = waveform.shape[-1] / target_rate

        # split up audio in chunks if necessary
        if length_seconds > chunk_size:
            nchunks = math.ceil(length_seconds / chunk_size)
        else:
            nchunks = 1

        if nchunks > 1:
            chunks = waveform.chunk(nchunks, dim=-1)
        else:
            chunks = [waveform]  # type: ignore

        # collect all chunk embeddings
        all_feats_list = []
        with torch.no_grad():
            for chunk in chunks:
                chunk = chunk.unsqueeze(0)  # add batch dimension: (1, T_chunk)

                outputs = model(chunk, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                feats = hidden_states[layer_n]  # (1, T_chunk, D)

                feat = feats.squeeze(0)  # drop batch dimension: (T_chunk, D)
                all_feats_list.append(feat.cpu())

        # concatenate along time axis: (T_total, D)
        full_feat = torch.cat(all_feats_list, dim=0).numpy()

        # save embedding
        output_path = feature_output_dir / sample_id
        np.savez_compressed(str(output_path), data=full_feat)

    print(f"Finished extracting layer {layer_n}")
