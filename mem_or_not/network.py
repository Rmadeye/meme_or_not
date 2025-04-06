import pytorch_lightning as pl
import torchvision.models as models
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torch


class MemeClassifier(pl.LightningModule):
    def __init__(self, dropout_rate=0.3):  # Możesz dostosować dropout_rate
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Dodajemy Dropout przed klasyfikacją
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Nowa warstwa Dropout
            nn.Linear(in_features, 2),  # Klasyfikacja na 2 klasy
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
        loss = self.loss_fn(y_hat, y)  # Loss expecting y as [batch_size]
        acc = self.accuracy(
            y_hat, y
        )  # Accuracy will also work with logits directly
        probabilities = torch.softmax(
            y_hat, dim=1
        )  # Zastosowanie softmax tylko w test_step
        self.log("test_acc", acc, prog_bar=True)
        return {
            "test_loss": loss,
            "test_acc": acc,
            "test_probabilities": probabilities,  # Zwracanie prawdopodobieństw
        }

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        probabilities = torch.softmax(
            y_hat, dim=1
        )  # Zastosowanie softmax dla predykcji
        return probabilities

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
