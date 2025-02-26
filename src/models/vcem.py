import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.metrics import Task_Accuracy, Concept_Accuracy
from src.utilities import get_intervened_concepts_predictions
from torchvision.models import resnet34

class VariationalConceptEmbeddingModel(pl.LightningModule):
    def __init__(self, 
                 in_size, 
                 n_concepts, 
                 n_classes,
                 emb_size, 
                 task_penalty,
                 kl_penalty,
                 p_int_train=None,
                 train_backbone=False,
                 sampling=False):
        super().__init__()

        self.in_size = in_size
        self.n_concepts = n_concepts
        self.emb_size = emb_size
        self.task_penalty = task_penalty
        self.kl_penalty = kl_penalty
        self.n_classes = n_classes
        self.has_concepts = True
        self.task_metric = Task_Accuracy()
        self.concept_metric = Concept_Accuracy()
        self.sampling = sampling
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

        self.train_backbone = train_backbone
        if self.train_backbone:
            self.setup_backbone()

    def setup_backbone(self):
        self.backbone = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # freeze all the layers except the last one
        for param in self.backbone.parameters():
            param.requires_grad = True
        # Unfreeze the last layer
        #for param in self.backbone[-2].parameters():
        #    param.requires_grad = True
        #for param in self.backbone[-1].parameters():
        #    param.requires_grad = True
        print('Backbone setup done!')

    def apply_intervention(self, c_pred, c_int, c_emb, concept_idx):
        c_int = c_int.unsqueeze(-1).expand(-1, -1, self.prototype_emb_pos.shape[-1]).int()
        cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, self.prototype_emb_pos.shape[-1])
        cloned_c_pred = torch.where(cloned_c_pred>0.5,1,0)
        prototype_emb = cloned_c_pred * self.prototype_emb_pos[None, concept_idx, :] + (1 - cloned_c_pred) * self.prototype_emb_neg[None, concept_idx, :]
        prototype_emb = prototype_emb.squeeze()
        c_int = c_int.squeeze()
        int_emb = c_int * prototype_emb + (1 - c_int) * c_emb
        return int_emb

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c, noise=None, p_int=None):
        bsz = x.shape[0]
        p_int = self.p_int_train if (self.training and self.current_epoch>25) else p_int
        if self.train_backbone:
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
        if noise!=None:
            eps = torch.randn_like(x)
            x = eps * noise + x * (1-noise)
        c_pred_list, c_emb_list, mu_list, logvar_list = [], [], [], []
        x = self.shared_layers(x)
        for i in range(self.n_concepts):
            c_pred = self.concept_scorers[i](x) 
            emb = self.layers[i](torch.cat([x, c_pred], dim=-1))
            mu = self.mu_layer[i](emb)
            logvar = self.logvar_layer[i](emb)
            if self.training and self.sampling:
                c_emb = self.reparameterize(mu, logvar)
            else:
                c_emb = mu
            if p_int!=None:
                int_mask, c_pred = get_intervened_concepts_predictions(c_pred, c[:,i].unsqueeze(-1), p_int, True, self.training)
                c_emb = self.apply_intervention(c_pred, int_mask, c_emb, i)
            c_emb_list.append(c_emb.unsqueeze(1))
            c_pred_list.append(c_pred.unsqueeze(1))
            mu_list.append(mu.unsqueeze(1))
            logvar_list.append(logvar.unsqueeze(1))
        c_emb = torch.cat(c_emb_list, dim=1) 
        c_pred = torch.cat(c_pred_list, dim=1)[:,:,0] 
        mu = torch.cat(mu_list, dim=1) 
        logvar = torch.cat(logvar_list, dim=1)
        y_pred = self.classifier(c_emb.view(bsz, -1))
        return c_pred, y_pred, c_emb, mu, logvar

    def D_kl_gaussian(self, mu_q, logvar_q, mu_p):
        if self.sampling:
            dot_prod = torch.bmm((mu_q - mu_p), (mu_q - mu_p).permute(0,2,1)).diagonal(dim1=-2, dim2=-1)
            d_kl = 0.5 * (dot_prod - self.emb_size - logvar_q.sum(dim=-1) + logvar_q.exp().sum(dim=-1))    
        else:
            d_kl = torch.bmm((mu_q - mu_p), (mu_q - mu_p).permute(0,2,1)).diagonal(dim1=-2, dim2=-1)
        return d_kl.mean() # average over the batch

    def step(self, batch, batch_idx, noise=None, p_int=None):
        x, concept_labels, y = batch
        c_pred, y_pred, c_emb, mu, logvar = self.forward(x, concept_labels, noise, p_int)
        concept_loss, task_loss, D_kl = self.compute_losses(y_pred, c_pred, mu, logvar, concept_labels, y)
        return task_loss, concept_loss, D_kl, c_pred, y_pred, c_emb, mu, logvar, concept_labels, y

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
        prototype_emb = cloned_c_pred * self.prototype_emb_pos[None, :, :] + (1 - cloned_c_pred) * self.prototype_emb_neg[None, :, :]
        D_kl = self.D_kl_gaussian(mu, logvar, prototype_emb)
        return concept_loss, task_loss, D_kl

    def training_step(self, batch, batch_idx):
        task_loss, concept_loss, D_kl, _, _, _, _, _, _, _ = self.step(batch, batch_idx)

        self.log('train_concept_loss', concept_loss)
        self.log('train_task_loss', self.task_penalty*task_loss)
        self.log('train_kl_loss', self.kl_penalty*D_kl)

        loss = concept_loss + (task_loss * self.task_penalty) + (D_kl * self.kl_penalty)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        task_loss, concept_loss, D_kl, _, _, _, _, _, _, _ = self.step(batch, batch_idx)

        self.log('val_concept_loss', concept_loss)
        self.log('val_task_loss', self.task_penalty*task_loss)
        self.log('val_kl_loss', self.kl_penalty*D_kl)
        
        loss = concept_loss + (task_loss * self.task_penalty) + (D_kl * self.kl_penalty)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        task_loss, concept_loss, D_kl, c_pred, y_pred, _, _, _, c, y = self.step(batch, batch_idx)

        task_acc = self.task_metric(y_pred, y)
        self.log('test_task_acc', task_acc)

        concept_acc = self.concept_metric(c_pred, c)
        self.log('test_concept_acc', concept_acc)
        
        loss = concept_loss + (task_loss * self.task_penalty) + (D_kl * self.kl_penalty)

        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
