import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.metrics import Task_Accuracy, Concept_Accuracy
from src.utilities import get_intervened_concepts_predictions

class ConceptEmbeddingModel(pl.LightningModule):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            n_classes,
            task_penalty,
            p_int_train=0.25,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.task_penalty = task_penalty
        self.p_int_train = p_int_train
        self.n_classes = n_classes
        self.n_concepts = n_concepts
        self.has_concepts = True
        self.task_metric = Task_Accuracy()
        self.concept_metric = Concept_Accuracy()

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(self.n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
            torch.nn.Sigmoid(),
        )

       # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_size*self.n_concepts, self.n_concepts),
            nn.ReLU(),
            nn.Linear(self.n_concepts, self.n_classes)
        )

    def forward(self, x, c, noise=None, p_int=None):
        if noise!=None:
            eps = torch.randn_like(x)
            x = eps * noise + x * (1-noise)
        p_int = self.p_int_train if self.training else p_int
        c_emb_list, c_pred_list = [], []
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            c_pred_list.append(c_pred)
            if p_int!=None:
                c_pred = get_intervened_concepts_predictions(c_pred, c[:,i].unsqueeze(-1), p_int)
            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))
        c_embs = torch.cat(c_emb_list, axis=1)
        c_preds = torch.cat(c_pred_list, axis=1)
        y_hat = self.classifier(c_embs.flatten(start_dim=1))
        return c_preds, y_hat, c_embs

    def step(self, batch, batch_idx, noise=None, p_int=None):
        x, c, y = batch
        c_pred, y_hat, c_embs = self.forward(x, c, noise, p_int)
        task_loss = F.cross_entropy(y_hat, y)
        concept_loss = 0
        for i in range(c.shape[1]):
            concept_loss += F.binary_cross_entropy(c_pred[:,i], c[:,i])
        concept_loss /= c.shape[1]
        loss = concept_loss + task_loss * self.task_penalty
        return loss, task_loss, concept_loss, c, y, c_pred, y_hat, c_embs

    def training_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, _, _, _, _, _ = self.step(batch, batch_idx)
        self.log('train_task_loss', task_loss*self.task_penalty)
        self.log('train_concept_loss', concept_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, _, _, _, _, _ = self.step(batch, batch_idx)
        self.log('val_task_loss', task_loss*self.task_penalty)
        self.log('val_concept_loss', concept_loss)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, task_loss, concept_loss, c, y, c_pred, y_hat, _ = self.step(batch, batch_idx)
        self.log('test_task_loss', task_loss*self.task_penalty)
        self.log('test_concept_loss', concept_loss)
        self.log('test_loss', loss)

        task_acc = self.task_metric(y_hat, y)
        self.log('test_task_acc', task_acc)

        concept_acc = self.concept_metric(c_pred, c)
        self.log('test_concept_acc', concept_acc)
        return loss
    
    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
    

