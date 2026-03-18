import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleCNN(nn.Module):
    def __init__(self, n_hidden_layers, n_kernels, kernel_size):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        layers = [
            nn.Conv2d(1, n_kernels, kernel_size=kernel_size, padding='same'),
            nn.GroupNorm(4, n_kernels),
            nn.PReLU()
        ]

        for _ in range(self.n_hidden_layers):
            layers.extend([
                nn.Conv2d(n_kernels, n_kernels, kernel_size=kernel_size, padding='same'),
                nn.GroupNorm(4, n_kernels),
                nn.PReLU(),
            ])

        layers.extend([
            nn.Conv2d(n_kernels, 1, kernel_size=1),
            nn.Sigmoid()
        ])

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layers(x)

class MicrographCleaner(pl.LightningModule):
    def __init__(self, n_hidden_layers=12, n_kernels=16, kernel_size=5, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleCNN(n_hidden_layers, n_kernels, kernel_size)
        self.lossF = nn.BCELoss()
        self.learning_rate = learning_rate
        self.val_imgs_to_log = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.lossF(outputs, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.lossF(outputs, masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)