"""Train script."""

import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from birdclef.datamodules.datamodule import BirdCLEFDataModule
from birdclef.datasets.birdclef2024 import BirdCLEF2024Dataset
from birdclef.modeling.encodec.model import EncodecClassifier
from birdclef.trainer import ClassifierLightningModule

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
    parser.add_argument("--num_devices", default=1, type=int)
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overfit", action="store_true")

    args = parser.parse_args()

    seed_everything(1234)

    log_path = args.log_path / args.name
    log_path.mkdir(exist_ok=True, parents=True)

    model = EncodecClassifier()
    pl_module = ClassifierLightningModule(model)

    dataset = BirdCLEF2024Dataset(args.data_root)
    datamodule = BirdCLEFDataModule(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path), strict=True)

    logger = WandbLogger(
        project="birdclef-2024", save_dir=log_path, name=args.name, offline=args.debug
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename="{step}",
        save_last=True,
        monitor="step",
        save_top_k=5,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = log_path / "last.ckpt"

    trainer = Trainer(
        default_root_dir=log_path,
        max_epochs=1000,
        accelerator="auto",
        devices=args.num_devices,
        logger=logger,
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_callback],
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        overfit_batches=0.05 if args.overfit else 0.0,
        detect_anomaly=False,
        strategy="ddp_find_unused_parameters_true" if args.num_devices > 1 else "auto",
    )
    trainer.fit(
        pl_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )


if __name__ == "__main__":
    main()
