import torch
import torch.nn as nn
import pytorch_lightning as pl


# ------------------------------------------------------------------ #
# Building blocks
# ------------------------------------------------------------------ #

class ResBlock(nn.Module):
    """Residual block with GroupNorm + dropout."""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class EncoderBlock(nn.Module):
    """Conv to change channels + 2 residual blocks."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(out_ch, dropout)
        self.res2 = ResBlock(out_ch, dropout)

    def forward(self, x):
        x = self.entry(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class DecoderBlock(nn.Module):
    """Upsample + concat skip + 2 residual blocks."""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.entry = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(out_ch, dropout)
        self.res2 = ResBlock(out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        x = self.entry(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — captures multi-scale context."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6,  bias=False), nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False), nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False), nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True))
        self.pool  = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.ReLU(inplace=True))
        self.proj  = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1))

    def forward(self, x):
        p = self.pool(x)
        p = nn.functional.interpolate(p, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.proj(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x), p], dim=1))


# ------------------------------------------------------------------ #
# StrongUNet
# ------------------------------------------------------------------ #

class StrongUNet(nn.Module):
    """
    Deep Residual UNet with ASPP bottleneck.
    base=64 → ~31M params. Designed for cryo-EM segmentation.
    Input:  (B, 1, H, W)
    Output: (B, 1, H, W) in [0, 1]
    """
    def __init__(self, base=64):
        super().__init__()
        b = base  # 64

        # Encoder
        self.enc1 = EncoderBlock(1,    b,    dropout=0.05)   # 64
        self.enc2 = EncoderBlock(b,    b*2,  dropout=0.1)    # 128
        self.enc3 = EncoderBlock(b*2,  b*4,  dropout=0.1)    # 256
        self.enc4 = EncoderBlock(b*4,  b*8,  dropout=0.15)   # 512

        self.pool = nn.MaxPool2d(2)

        # Bottleneck with ASPP
        self.bottleneck = nn.Sequential(
            EncoderBlock(b*8, b*16, dropout=0.2),             # 1024
            ASPP(b*16, b*16),
        )

        # Decoder
        self.dec4 = DecoderBlock(b*16, b*8,  b*8,  dropout=0.15)
        self.dec3 = DecoderBlock(b*8,  b*4,  b*4,  dropout=0.1)
        self.dec2 = DecoderBlock(b*4,  b*2,  b*2,  dropout=0.1)
        self.dec1 = DecoderBlock(b*2,  b,    b,    dropout=0.05)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(b, b//2, 3, padding=1, bias=False),
            nn.GroupNorm(8, b//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(b//2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        x  = self.dec4(b,  e4)
        x  = self.dec3(x,  e3)
        x  = self.dec2(x,  e2)
        x  = self.dec1(x,  e1)
        return self.head(x)


# ------------------------------------------------------------------ #
# Loss: BCE + Dice (Dice-heavy for cryo-EM)
# ------------------------------------------------------------------ #

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target, smooth=1.0):
        p = pred.view(-1)
        t = target.view(-1)
        return 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + (1 - self.bce_weight) * self.dice_loss(pred, target)


# ------------------------------------------------------------------ #
# Lightning wrapper — compatible with existing train.py / infer.py
# ------------------------------------------------------------------ #

class MicrographCleaner(pl.LightningModule):
    def __init__(self, base=64, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = StrongUNet(base=base)
        self.lossF = BCEDiceLoss(bce_weight=0.3)
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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=45,
            steps_per_epoch=100,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }
