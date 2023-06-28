from lit_classification import LitClassifier
from data import FlowerDataset
import os
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn as nn
from clearml import Task, Dataset
import timm

task = Task.init(
    project_name="Mlops Demo",
    task_name="flower classification",
    tags="development",
)
seed_everything(42)


def get_args_parser():
    parser = argparse.ArgumentParser("Set trimming detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--desc", type=str, default="")
    parser.add_argument("--accumulate", type=int, default=1)
    return parser


def make_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"train_logs/{args.model}/checkpoints",
        monitor="val_loss",
        mode="min",
        filename="best",
    )
    lr_callback = LearningRateMonitor("step")
    return [lr_callback, checkpoint_callback]


def main(args):
    model = timm.create_model(args.model, pretrained=True, num_classes=args.num_classes)

    L = LitClassifier(model, args)
    root_folder = Dataset.get(
        dataset_name="flower", dataset_project="Mlops Demo"
    ).get_local_copy()
    train_dataset = FlowerDataset(
        root_folder=root_folder,
        img_size=(args.img_size, args.img_size),
        is_training=True,
        model=model,
    )
    val_dataset = FlowerDataset(
        root_folder=root_folder,
        img_size=(args.img_size, args.img_size),
        is_training=False,
        model=model,
    )
    print("Number of train images:", len(train_dataset))
    print("Number of validate images:", len(val_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8)

    callbacks = make_callbacks(args)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.device],
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate,
    )
    trainer.fit(model=L, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Flower Classification", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
