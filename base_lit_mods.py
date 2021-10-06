from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn
from pytorch_lightning.metrics import Accuracy


class BaseClassifierLitMod(LightningModule):
    def __init__(self, lr=1e-3, weight_decay=0.0):
        super().__init__()

        self.lr = lr
        self.lr = 2e-5
        self.wd = weight_decay

        self.accuracy = Accuracy()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return {
           'optimizer': optimizer,
           'monitor': 'val/loss'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        accuracy = self.accuracy(nn.functional.softmax(logits, dim=1), y)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)
        return loss