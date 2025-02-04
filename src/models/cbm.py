import pytorch_lightning as pl
import torch

import torch.nn as nn
import torch.nn.functional as F
from src.metrics import Task_Accuracy, Concept_Accuracy

class ConceptBottleneckModel(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 n_concepts, 
                 n_labels,
                 task_interpretable=True):
        super(ConceptBottleneckModel, self).__init__()

        # Encoder: Maps input to concepts
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_concepts),
            nn.Sigmoid()
        )

        # Decoder: Maps concepts to output
        if task_interpretable:
            self.decoder = nn.Sequential(
                nn.Linear(n_concepts, n_labels)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(n_concepts, n_concepts),
                nn.ReLU(),
                nn.Linear(n_concepts, n_labels)
            )

        self.has_concepts = True
        self.task_metric = Task_Accuracy()
        self.concept_metric = Concept_Accuracy()
        
    def forward(self, x):
        concepts = self.encoder(x)
        output = self.decoder(concepts)
        return concepts, output

    def step(self, batch, batch_idx):
        x, c, y = batch
        c_pred, y_hat = self.forward(x)
        task_loss = F.cross_entropy(y_hat, y)
        concept_loss = 0
        for i in range(c.shape[1]):
            concept_loss += F.binary_cross_entropy(c_pred[:,i], c[:,i])
        concept_loss /= c.shape[1]
        loss = concept_loss + task_loss
        return loss, task_loss, concept_loss, c, y, c_pred, y_hat

    def training_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, _, _, _, _ = self.step(batch, batch_idx)
        self.log('train_label_loss', task_loss)
        self.log('train_concept_loss', concept_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, _, _, _, _ = self.step(batch, batch_idx)
        self.log('val_label_loss', task_loss)
        self.log('val_concept_loss', concept_loss)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, c, y, c_pred, y_hat = self.step(batch, batch_idx)
        self.log('test_label_loss', task_loss)
        self.log('test_concept_loss', concept_loss)
        self.log('test_loss', loss)

        task_acc = self.task_metric(y_hat, y)
        self.log('test_task_acc', task_acc)

        concept_acc = self.concept_metric(c_pred, c)
        self.log('test_concept_acc', concept_acc)
        return loss
    
    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
