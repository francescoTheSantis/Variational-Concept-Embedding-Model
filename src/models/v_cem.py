import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.utilities import get_intervened_concepts_predictions, D_kl_gaussian

class V_CEM(pl.LightningModule):
    def __init__(self, 
                 in_size, 
                 n_concepts, 
                 n_classes,
                 emb_size, 
                 task_penalty,
                 kl_penalty,
                 p_int_train=0.1):
        super().__init__()

        self.in_size = in_size
        self.n_concepts = n_concepts
        self.emb_size = emb_size
        self.task_penalty = task_penalty
        self.kl_penalty = kl_penalty
        self.n_classes = n_classes
        self.p_int_train = p_int_train

        # Initialize learnable concept prototypes using normal distribution
        self.prototype_emb_pos = nn.Parameter(torch.randn(n_concepts, emb_size))
        self.prototype_emb_neg = nn.Parameter(torch.randn(n_concepts, emb_size))

        self.shared_layers = nn.Sequential(
            nn.Linear(self.in_size, self.in_size), 
            nn.ReLU()
        )

        self.concept_scorers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.mu_layer = nn.ModuleList()
        self.logvar_layer = nn.ModuleList()

        for _ in range(self.n_concepts):
            layers = nn.Sequential(
                nn.Linear(self.in_size, 1),
                nn.Sigmoid()
            )
            self.concept_scorers.append(layers)

            layers = nn.Sequential(
                nn.Linear(self.in_size + 1, self.in_size + 1),
                nn.ReLU()
            )
            self.layers.append(layers)

            self.mu_layer.append(nn.Linear(self.in_size + 1, self.emb_size))  
            self.logvar_layer.append(nn.Linear(self.in_size + 1, self.emb_size))

        self.classifier = nn.Sequential(
            nn.Linear(self.emb_size*self.n_concepts, self.n_concepts),
            nn.ReLU(),
            nn.Linear(self.n_concepts, self.n_classes)
        )

    def apply_intervention(self, c_pred, c_int, c_emb, concept_idx):
        c_int = c_int.unsqueeze(-1).expand(-1, -1, self.prototype_emb_pos.shape[-1])
        cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, self.prototype_emb_pos.shape[-1])
        cloned_c_pred = torch.where(cloned_c_pred>0.5,1,0)
        prototype_emb = cloned_c_pred * self.prototype_emb_pos[None, :, :] + (1 - cloned_c_pred) * self.prototype_emb_neg[None, :, :]
        prototype_emb = prototype_emb[:,concept_idx,:]
        c_int = c_int.squeeze()
        int_emb = c_int * prototype_emb + (1 - c_int) * c_emb
        return int_emb

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, c=None, p_int=None):
        bsz = x.shape[0]
        p_int = self.p_int_train if self.training else p_int
        c_pred_list, c_emb_list, mu_list, logvar_list = [], [], [], []
        x = self.shared_layers(x)
        for i in range(self.n_concepts):
            c_pred = self.concept_scorers[i](x) 
            if c!=None and p_int!=None and self.training:
                c_pred = get_intervened_concepts_predictions(c_pred, c[:,i].unsqueeze(-1), p_int, False)
            emb = self.layers[i](torch.cat([x, c_pred], dim=-1))
            mu = self.mu_layer[i](emb)
            logvar = self.logvar_layer[i](emb)
            # during training we sample from the multivariate normal distribution, at test-time we take MAP.
            if self.training:
                c_emb = self.reparameterize(mu, logvar)
            else:
                c_emb = mu
            # apply prototype interventions
            if c!=None and p_int!=None:
                # generate the mask containing one for the indexes that have to be intervened and 0 otherwise
                int_mask, c_pred = get_intervened_concepts_predictions(c_pred, c[:,i].unsqueeze(-1), p_int, True)
                c_emb = self.apply_intervention(c_pred, int_mask, c_emb, i)
            c_emb_list.append(c_emb.unsqueeze(1))
            c_pred_list.append(c_pred.unsqueeze(1))
            mu_list.append(mu.unsqueeze(1))
            logvar_list.append(logvar.unsqueeze(1))
        # join all the concepts by concatenating them along the second dimension
        c_emb = torch.cat(c_emb_list, dim=1) # (batch_size, n_concepts, emb_size)
        c_pred = torch.cat(c_pred_list, dim=1)[:,:,0] # (batch_size, n_concepts)
        mu = torch.cat(mu_list, dim=1) # (batch_size, n_concepts, emb_size)
        logvar = torch.cat(logvar_list, dim=1) # (batch_size, n_concepts, emb_size)

        y_pred = self.classifier(c_emb.view(bsz, -1))

        return y_pred, c_pred, c_emb, mu, logvar

    def D_kl_gaussian(self, mu_q, logvar_q, mu_p):
        value = -0.5 * torch.sum(1 + logvar_q - (mu_q - mu_p).pow(2) - logvar_q.exp(), dim=-1)
        return value.mean()

    def compute_losses(self, y_pred, c_pred, mu, logvar, c, y):
        concept_form = nn.BCELoss()
        task_form = nn.CrossEntropyLoss()

        # compute the concept loss and avergae over the number of concepts
        concept_loss = 0
        for i in range(self.n_concepts):
            concept_loss += concept_form(c_pred[:,i], c[:,i])
        concept_loss /= self.n_concepts 
    
        # task loss
        task_loss = task_form(y_pred, y.long())

        # KL divergence
        cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, self.emb_size)
        prototype_emb = cloned_c_pred * self.prototype_emb_pos[None, :, :] + \
            (1 - cloned_c_pred) * self.prototype_emb_neg[None, :, :]
        D_kl = D_kl_gaussian(mu, logvar, prototype_emb)
        
        return concept_loss, task_loss, D_kl

    def training_step(self, batch, batch_idx):
        x, concept_labels, y = batch
        y_pred, c_pred, _, mu, logvar = self(x, concept_labels)

        concept_loss, task_loss, D_kl = self.compute_losses(y_pred, c_pred, mu, logvar, concept_labels, y)

        self.log('train_concept_loss', concept_loss)
        self.log('train_task_loss', task_loss)
        self.log('train_kl_loss', D_kl)
        
        loss = concept_loss + (task_loss * self.task_penalty) + (D_kl * self.kl_penalty)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, concept_labels, y = batch
        y_pred, c_pred, _, mu, logvar = self(x, concept_labels)
        print('Am i sampling?', self.training)

        concept_loss, task_loss, D_kl = self.compute_losses(y_pred, c_pred, mu, logvar, concept_labels, y)

        self.log('val_concept_loss', concept_loss)
        self.log('val_task_loss', task_loss)
        self.log('val_kl_loss', D_kl)
        
        loss = concept_loss + (task_loss * self.task_penalty) + (D_kl * self.kl_penalty)
        return loss

    def test_step(self, batch, batch_idx):
        x, concept_labels, y = batch
        y_pred, c_pred, _, mu, logvar = self(x, concept_labels)

        concept_loss, task_loss, D_kl = self.compute_losses(y_pred, c_pred, mu, logvar, concept_labels, y)

        self.log('test_concept_loss', concept_loss)
        self.log('test_task_loss', task_loss)
        self.log('test_kl_loss', D_kl)
        
        loss = concept_loss + (task_loss * self.task_penalty) + (D_kl * self.kl_penalty)
        return loss

    def predict(self, x):
        y_pred, c_pred, c_emb, mu, logvar = self(x)
        return y_pred, c_pred, c_emb, mu, logvar
