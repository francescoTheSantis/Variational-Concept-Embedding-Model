import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch
from torch.optim import Adam
import numpy as np
import pandas as pd
from src.metrics import f1_acc_metrics
from tqdm import tqdm

class Trainer:
    def __init__(self, model, cfg, wandb_logger, csv_logger):
        self.cfg = cfg
        self.wandb_logger = wandb_logger
        self.csv_logger = csv_logger
        self.model = model
        self.epss = np.arange(0, 1.1, 0.1)
        self.p_ints = np.arange(0, 1.1, 0.1)

    def build_trainer(self):
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=self.cfg.patience, 
            verbose=True, 
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss', 
            filename='best_model', 
            save_top_k=1, 
            mode='min', 
            verbose=True
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        self.trainer = pl.Trainer(
            max_epochs=self.cfg.max_epochs,
            callbacks=[early_stopping, checkpoint_callback, lr_monitor],
            logger=[self.wandb_logger, self.csv_logger],
            devices=self.cfg.gpus,  
            accelerator="gpu" 
        )

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.dataset.metadata.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.lr_step, gamma=self.cfg.gamma, verbose=True)

        # Set the optimizer in the repsective model
        self.model.optimizer = self.optimizer
        self.model.scheduler = self.scheduler

    def train(self, train_dataloader, val_dataloader):
        self.trainer.fit(self.model, 
                         train_dataloader, 
                         val_dataloader)
    
    def test(self, test_dataloader):
        self.trainer.test(self.model, test_dataloader)

    def interventions(self, test_dataloader):
        intervention_df = pd.DataFrame(columns=['noise', 'p_int', 'f1', 'accuracy'])
        self.model.eval()
        with torch.no_grad():
            for eps in self.epss:
                print('Performing interventions with noise:', eps)
                for p_int in tqdm(self.p_ints):
                    y_preds = []
                    y_trues = []
                    for batch in test_dataloader:
                        x, c, y = batch
                        output = self.model.forward(x, c, eps, p_int)
                        y_pred = output[1]
                        y_preds.append(y_pred)
                        y_trues.append(y)
                    y = torch.cat(y_trues, dim=0)
                    y_preds = torch.cat(y_preds, dim=0)
                    y = y.cpu().numpy()
                    y_preds = y_preds.argmax(-1).cpu().numpy()
                    task_f1, task_acc = f1_acc_metrics(y, y_preds)
                    intervention_results = {'noise': round(eps,1), 'p_int': round(p_int,1), 'f1': round(task_f1,2), 'accuracy': round(task_acc,2)}
                    intervention_df = pd.concat([intervention_df, pd.DataFrame([intervention_results])], ignore_index=True)
        return intervention_df
    
    def get_latents(self, test_dataloader):
        latents = []
        concept_ground_truth = []
        labels = []
        if self.model.__class__.__name__ == 'ConceptBottleneckModel':
            latent_concept_idx = 5
            true_concept_idx = 3
            true_label_idx = 4
        elif self.model.__class__.__name__ == 'ConceptEmbeddingModel':
            latent_concept_idx = 7
            true_concept_idx = 3
            true_label_idx = 4
        elif self.model.__class__.__name__ == 'VariationalConceptEmbeddingModel':
            latent_concept_idx = 5
            true_concept_idx = 8
            true_label_idx = 9
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(test_dataloader):
                output = self.model.step(batch, batch_id)
                latents.append(output[latent_concept_idx]) # concept representation
                concept_ground_truth.append(output[true_concept_idx]) # real concepts
                labels.append(output[true_label_idx]) # real labels
        latents = torch.cat(latents, dim=0)
        concept_ground_truth = torch.cat(concept_ground_truth, dim=0)
        labels = torch.cat(labels, dim=0)
        if self.model.__class__.__name__ in ['VariationalConceptEmbeddingModel', 'ConceptEmbeddingModel']:
            latents = latents.flatten(start_dim=1)
        return latents, concept_ground_truth, labels

