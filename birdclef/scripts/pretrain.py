"""Train script."""

import argparse
from pathlib import Path

import torch

from birdclef.datasets.unlabeled import UnlabeledDataset
from birdclef.modeling import ConformerModelForPreTraining
from birdclef.trainers.pretrainer import PreTrainer
from birdclef.utils import seed_everything

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

MODELS = {"conformer": ConformerModelForPreTraining}


def main() -> None:
    """Train script."""
    parser = argparse.ArgumentParser(description="Train script.")
    parser.add_argument("model", type=str)
    parser.add_argument("name", type=str)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--log_root", default="logs", type=Path)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cross_validate", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)

    log_path = args.log_root / args.name
    log_path.mkdir(exist_ok=True, parents=True)

    dataset = UnlabeledDataset(args.data_root)

    if args.model not in MODELS:
        raise ValueError(f"Invalid model: {args.model}")

    model = MODELS[args.model]()

    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path), strict=True)

    trainer = PreTrainer(
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
