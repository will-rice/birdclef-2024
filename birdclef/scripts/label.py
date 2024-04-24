"""Pseudo-labeling script for BirdCLEF 2024."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

SAMPLE_RATE = 32000
MAX_LENGTH = 5 * SAMPLE_RATE


def main() -> None:
    """Label dataset."""
    parser = argparse.ArgumentParser(description="Label dataset.")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("target_root", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    train_labels = sorted(
        {
            p.parent.name
            for p in Path("/data-fast/birdclef-2024/train_audio").glob("**/*.ogg")
        }
    )
    source_root = args.source_root
    target_root = args.target_root
    target_root.mkdir(exist_ok=True, parents=True)
    models = [load_model(path) for path in args.model_root.glob("*.pt")]
    paths = sorted(source_root.glob("**/*.ogg"))
    metadata = []

    def predict(path: Path) -> None:
        """Predict labels for audio file."""
        audio, sr = torchaudio.load(path)
        for i in range(0, audio.shape[-1] // 32000, 5):
            chunk_end_time = i + 5
            start_idx = i * sr
            audio_chunk = audio[..., start_idx : start_idx + MAX_LENGTH]
            with torch.jit.optimized_execution(False):
                with torch.no_grad():
                    probs = torch.zeros(len(train_labels))
                    for model in models:
                        probs += model(audio_chunk[None]).squeeze().sigmoid()
                    probs /= len(models)
            _, label_ids = probs.topk(4)
            labels = [
                train_labels[label_id.item()]
                for label_id in label_ids
                if probs[label_id] > 0.5
            ]
            if labels:
                primary_label = labels[0]
            elif probs.max() < 0.1:
                primary_label = "nocall"
            else:
                continue

            secondary_labels = labels[1:]
            save_path = (
                target_root / primary_label / f"{path.stem}_{chunk_end_time}.flac"
            )
            save_path.parent.mkdir(exist_ok=True, parents=True)
            torchaudio.save(str(save_path), audio_chunk, sr)
            metadata.append(
                {
                    "primary_label": primary_label,
                    "secondary_labels": str(secondary_labels),
                    "filename": save_path.relative_to(target_root),
                }
            )

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        _ = list(tqdm(executor.map(predict, paths), total=len(paths)))

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(target_root / "metadata.csv", index=False)


def load_model(model_path: Path) -> Any:
    """Load model."""
    model = torch.jit.load(model_path, map_location="cpu").eval()
    model = torch.compile(model, fullgraph=True, mode="max-autotune-no-cudagraphs")
    return model


if __name__ == "__main__":
    main()
