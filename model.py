import torch
import torch.nn as nn
import pytorch_lightning as pl


# ------------------------------------------------------------------ #
# Residual double conv block
# ------------------------------------------------------------------ #

class ResDoubleConv(nn.Module):
    """Two conv layers with a residual (skip) connection inside the block."""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        # 1x1 conv to match channels if needed
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------ #
# Deep Residual UNet
# ------------------------------------------------------------------ #

class UNet(nn.Module):
    """
    Deep Residual UNet — 5 levels, residual blocks, dropout.
    Input:  (B, 1, H, W)
    Output: (B, 1, H, W) in [0, 1]
    """
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels  # 32

        # Encoder (5 levels)
        self.inc    = ResDoubleConv(1,    b)       # 1   → 32
        self.down1  = Down(b,    b*2)              # 32  → 64
        self.down2  = Down(b*2,  b*4)              # 64  → 128
        self.down3  = Down(b*4,  b*8)              # 128 → 256
        self.down4  = Down(b*8,  b*16)             # 256 → 512

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(b*16, b*32),             # 512 → 1024
            ResDoubleConv(b*32, b*16),             # 1024 → 512 (compress back)
            nn.ConvTranspose2d(b*16, b*16, kernel_size=2, stride=2),
        )

        # Decoder (5 levels)
        self.up1 = Up(b*32, b*8)                   # 512+512 → 256
        self.up2 = Up(b*16, b*4)                   # 256+256 → 128
        self.up3 = Up(b*8,  b*2)                   # 128+128 → 64
        self.up4 = Up(b*4,  b)                     # 64+64   → 32

        # Output
        self.outc = nn.Sequential(
            nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5b = self.bottleneck(x5)
        x  = self.up1(x5b, x5)
        x  = self.up2(x,   x4)
        x  = self.up3(x,   x3)
        x  = self.up4(x,   x2)
        return self.outc(x)


# ------------------------------------------------------------------ #
# Loss: BCE + Dice
# ------------------------------------------------------------------ #

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.4):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred_flat    = pred.view(-1)
        target_flat  = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + (1 - self.bce_weight) * self.dice_loss(pred, target)


# ------------------------------------------------------------------ #
# Lightning wrapper
# ------------------------------------------------------------------ #

class MicrographCleaner(pl.LightningModule):
    def __init__(self, base_channels=32, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model  = UNet(base_channels=base_channels)
        self.lossF  = BCEDiceLoss(bce_weight=0.4)
        self.learning_rate = learning_rate

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
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
