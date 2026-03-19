import torch
import torch.nn as nn
import pytorch_lightning as pl
 
 
# ------------------------------------------------------------------ #
# Building blocks
# ------------------------------------------------------------------ #
 
class DoubleConv(nn.Module):
    """Two conv layers each followed by GroupNorm + ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.block(x)
 
 
class Down(nn.Module):
    """Downsample with MaxPool then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
 
    def forward(self, x):
        return self.block(x)
 
 
class Up(nn.Module):
    """Upsample then concatenate skip connection then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd spatial dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
 
 
# ------------------------------------------------------------------ #
# UNet
# ------------------------------------------------------------------ #
 
class UNet(nn.Module):
    """
    Standard UNet with 4 downsampling levels.
    Input:  (B, 1, H, W)  — single-channel cryo-EM image
    Output: (B, 1, H, W)  — segmentation mask in [0, 1]
    """
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels  # 32 by default
 
        # Encoder
        self.inc   = DoubleConv(1,    b)      # 1  → 32
        self.down1 = Down(b,    b*2)          # 32 → 64
        self.down2 = Down(b*2,  b*4)          # 64 → 128
        self.down3 = Down(b*4,  b*8)          # 128→ 256
 
        # Bottleneck
        self.bottleneck = Down(b*8, b*16)     # 256→ 512
 
        # Decoder
        self.up1 = Up(b*16, b*8)             # 512→ 256
        self.up2 = Up(b*8,  b*4)             # 256→ 128
        self.up3 = Up(b*4,  b*2)             # 128→  64
        self.up4 = Up(b*2,  b)               #  64→  32
 
        # Output
        self.outc = nn.Sequential(
            nn.Conv2d(b, 1, kernel_size=1),
            nn.Sigmoid(),
        )
 
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)
 
 
# ------------------------------------------------------------------ #
# Loss: BCE + Dice
# ------------------------------------------------------------------ #
 
class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    Dice focuses on overlap — better for unbalanced segmentation masks.
    """
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()
 
    def dice_loss(self, pred, target, smooth=1.0):
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
 
    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + (1 - self.bce_weight) * self.dice_loss(pred, target)
 
 
# ------------------------------------------------------------------ #
# Lightning wrapper — drop-in replacement for MicrographCleaner
# ------------------------------------------------------------------ #
 
class MicrographCleaner(pl.LightningModule):
    def __init__(self, base_channels=32, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model   = UNet(base_channels=base_channels)
        self.lossF   = BCEDiceLoss(bce_weight=0.5)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},}
