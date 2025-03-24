import pytorch_lightning as pl
import torchvision.models as models
import torchmetrics
import torch.nn as nn
import torch.optim as optim


class MemeClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
