import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from src.metrics import Task_Accuracy

class BlackboxModel(pl.LightningModule):
    def __init__(self, input_dim, n_labels):
        super(BlackboxModel, self).__init__()
        # the hidden dimension is half of the input dimension
        hidden_dim = input_dim // 2
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, n_labels)
        self.has_concepts = False
        self.task_metric = Task_Accuracy()

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
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

        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]