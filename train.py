#!/usr/bin/env python3
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import MicrographCleaner
from dataset import TrainMicrographDataset, ValidationMicrographDataset


def main():
    # Training parameters
    WINDOW_SIZE = 512
    BATCH_SIZE = 2
    N_EPOCHS = 30

    # Load and split data
    train_df = pd.read_csv('train.csv')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = TrainMicrographDataset(train_df, window_size=WINDOW_SIZE)
    val_dataset = ValidationMicrographDataset(val_df, window_size=WINDOW_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # Initialize model
    model = MicrographCleaner()

    # Setup training
    logger = TensorBoardLogger('lightning_logs', name='micrograph_cleaner')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='micrograph-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator='auto',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save final checkpoint as final_checkpoint.pt
    trainer.save_checkpoint("final_checkpoint.pt")


if __name__ == "__main__":
    main()
