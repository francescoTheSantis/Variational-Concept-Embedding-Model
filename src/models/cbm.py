import pytorch_lightning as pl
import torch

import torch.nn as nn
import torch.nn.functional as F

class ConceptBottleneckModel(pl.LightningModule):
    def __init__(self, input_dim, concept_dim, output_dim, learning_rate=1e-3):
        super(ConceptBottleneckModel, self).__init__()
        self.learning_rate = learning_rate

        # Encoder: Maps input to concepts
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, concept_dim),
            nn.Sigmoid()
        )

        # Decoder: Maps concepts to output
        self.decoder = nn.Sequential(
            nn.Linear(concept_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        concepts = self.encoder(x)
        output = self.decoder(concepts)
        return concepts, output

    def training_step(self, batch, batch_idx):
        x, y = batch
        concepts, y_hat = self.forward(x)
        label_loss = F.cross_entropy(y_hat, y)
        concept_loss = F.binary_cross_entropy(concepts, x)
        loss = label_loss + concept_loss
        self.log('train_label_loss', label_loss)
        self.log('train_concept_loss', concept_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        concepts, y_hat = self.forward(x)
        label_loss = F.cross_entropy(y_hat, y)
        concept_loss = F.binary_cross_entropy(concepts, x)
        loss = label_loss + concept_loss
        self.log('val_label_loss', label_loss)
        self.log('val_concept_loss', concept_loss)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        concepts, y_hat = self.forward(x)
        label_loss = F.cross_entropy(y_hat, y)
        concept_loss = F.binary_cross_entropy(concepts, x)
        loss = label_loss + concept_loss
        self.log('test_label_loss', label_loss)
        self.log('test_concept_loss', concept_loss)
        self.log('test_loss', loss)
        return loss
