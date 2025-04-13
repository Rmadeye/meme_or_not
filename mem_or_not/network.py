import pytorch_lightning as pl
from sklearn.metrics import f1_score, precision_score, recall_score
import torchvision.models as models
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torch


class MemeClassifier(pl.LightningModule):
    def __init__(self, dropout_rate, lr):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.dropout_rate = float(dropout_rate)
        self.lr = float(lr)
        self.test_outputs = []
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features, 2),
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)  # Loss expecting y as [batch_size]
        acc = self.accuracy(
            y_hat, y
        )  # Accuracy will also work with logits directly
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)  # Loss expecting y as [batch_size]
        acc = self.accuracy(
            y_hat, y
        )  # Accuracy will also work with logits directly
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        proba = torch.softmax(y_hat, dim=1)
        preds = torch.argmax(proba, dim=1)

        acc = self.accuracy(preds, y)
        f1 = f1_score(
            y.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
            average="macro",
        )
        precision = precision_score(
            y.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
            average="macro",
        )
        recall = recall_score(
            y.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
            average="macro",
        )
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        self.test_outputs.append(
            {
                "loss": loss.detach().cpu(),
                "acc": acc.detach().cpu(),
                "y_true": y.detach().cpu(),
                "y_pred": preds.detach().cpu(),
                "y_proba": proba.detach().cpu(),
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

        return

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        probabilities = torch.softmax(y_hat, dim=1)
        return probabilities

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
