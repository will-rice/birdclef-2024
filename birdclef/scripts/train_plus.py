"""Train script."""

import argparse
from pathlib import Path
from typing import Any

import torch

from birdclef.datasets.birdclef2024plus import BirdCLEF2024DatasetPlus
from birdclef.modeling import (
    ConformerClassifier,
    Conv1DClassifier,
    Conv2DClassifier,
    ConvNextV2Classifier,
    EfficientFormerClassifier,
    EfficientNetClassifier,
    EfficientViTClassifier,
    EncodecClassifier,
    GeMClassifier,
    NFNetClassifier,
    SwiftFormerClassifier,
    WavLMClassifier,
)
from birdclef.trainers import StratifiedKFoldTrainer, Trainer
from birdclef.utils import seed_everything

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

MODELS = {
    "conv2d": Conv2DClassifier,
    "convnext": ConvNextV2Classifier,
    "efficientnet": EfficientNetClassifier,
    "efficientvit": EfficientViTClassifier,
    "efficientformer": EfficientFormerClassifier,
    "encodec": EncodecClassifier,
    "wavlm": WavLMClassifier,
    "conformer": ConformerClassifier,
    "nfnet": NFNetClassifier,
    "gem": GeMClassifier,
    "swiftformer": SwiftFormerClassifier,
    "conv1d": Conv1DClassifier,
}


def main() -> None:
    """Train script."""
    parser = argparse.ArgumentParser(description="Train script.")
    parser.add_argument("model", type=str)
    parser.add_argument("name", type=str)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--log_root", default="logs", type=Path)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cross_validate", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)

    log_path = args.log_root / args.name
    log_path.mkdir(exist_ok=True, parents=True)

    train_dataset = BirdCLEF2024DatasetPlus(
        args.data_root,
        augment=True,
        max_seconds=5,
        use_secondary_labels=True,
        random_crop=False,
        add_nocall=False,
    )
    val_dataset = BirdCLEF2024DatasetPlus(
        args.data_root,
        augment=False,
        max_seconds=5,
        use_secondary_labels=True,
        add_nocall=False,
    )

    if args.model not in MODELS:
        raise ValueError(f"Invalid model: {args.model}")

    model = MODELS[args.model](num_classes=train_dataset.num_classes)

    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path), strict=True)

    if args.cross_validate:
        trainer: Any = StratifiedKFoldTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            log_path=log_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            debug=args.debug,
            from_audio=args.model == "conv1d",
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            log_path=log_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            debug=args.debug,
            from_audio=args.model == "conv1d",
        )

    trainer.fit()


if __name__ == "__main__":
    main()
