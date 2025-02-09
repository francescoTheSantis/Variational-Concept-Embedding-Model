import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from src.metrics import Task_Accuracy
from torchvision.models import resnet34

class BlackboxModel(pl.LightningModule):
    def __init__(self, input_dim, n_labels, train_backbone=False):
        super().__init__()
        # the hidden dimension is half of the input dimension

        hidden_dim = input_dim // 2
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_labels)
        )

        self.has_concepts = False
        self.task_metric = Task_Accuracy()
        self.train_backbone = train_backbone

        if self.train_backbone:
            self.setup_backbone()

    def setup_backbone(self):
        self.backbone = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # freeze all the layers except the last one
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze the last layer
        for param in self.backbone[-2].parameters():
            param.requires_grad = True
        for param in self.backbone[-1].parameters():
            param.requires_grad = True
        print('Backbone setup done!')

    def forward(self, x):
        if self.train_backbone:
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
        x = self.model(x)
        return x

    def step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y.long())
        return loss, y, y_hat

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch, batch_idx)
        task_acc = self.task_metric(y_hat, y)
        self.log('test_task_acc', task_acc)
        self.log('test_concept_acc', torch.tensor(0.0))
        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
