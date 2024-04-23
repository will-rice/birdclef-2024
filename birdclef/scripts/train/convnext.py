"""Train script."""

import argparse
from pathlib import Path

import torch
from pytorch_lightning import seed_everything

from birdclef.datasets.birdclef2024 import BirdCLEF2024Dataset
from birdclef.kfold_trainer import StratifiedKFoldTrainer
from birdclef.modeling.convnext.model import ConvNextV2Classifier

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def main() -> None:
    """Train script."""
    parser = argparse.ArgumentParser(description="Train script.")
    parser.add_argument("name", type=str)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--log_path", default="logs", type=Path)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True, parents=True)

    dataset = BirdCLEF2024Dataset(args.data_root)

    model = ConvNextV2Classifier(num_classes=len(dataset.labels))

    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path), strict=True)

    trainer = StratifiedKFoldTrainer(
        model=model,
        dataset=dataset,
        log_path=log_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        debug=args.debug,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
