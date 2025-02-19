import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.metrics import Task_Accuracy, Concept_Accuracy
from src.utilities import get_intervened_concepts_predictions
from torchvision.models import resnet34

class ProbabilisticConceptBottleneckModel(pl.LightningModule):
    def __init__(self, 
                 in_size, 
                 n_concepts, 
                 n_classes,
                 emb_size, 
                 task_emb_size,
                 task_penalty,
                 kl_penalty,
                 p_int_train,
                 train_backbone=False,
                 mc_samples=1):
        super().__init__()

        self.in_size = in_size
        self.n_concepts = n_concepts
        self.emb_size = emb_size
        self.kl_penalty = kl_penalty
        self.n_classes = n_classes
        self.has_concepts = True
        self.task_metric = Task_Accuracy()
        self.concept_metric = Concept_Accuracy()
        self.p_int_train = p_int_train
        self.mc_samples = mc_samples
        self.task_emb_size = task_emb_size
        self.task_penalty = task_penalty

        # Initialize learnable concept prototypes using normal distribution
        self.prototype_emb_pos = nn.Parameter(torch.randn(n_concepts, emb_size))
        self.prototype_emb_neg = nn.Parameter(torch.randn(n_concepts, emb_size))

        # Task embeddings
        self.task_emb = nn.Parameter(torch.randn(n_classes, task_emb_size))

        # Initialize two learnable non negative parameters named "a" and "d"
        self.a = nn.Parameter(torch.ones(1)*5)
        self.d = nn.Parameter(torch.ones(1)*10)

        self.shared_layers = nn.Sequential(
            nn.Linear(self.in_size, self.in_size), 
            nn.ReLU()
        )

        self.fc = nn.Linear(self.emb_size*self.n_concepts, self.task_emb_size)

        self.mu_layer = nn.ModuleList()
        self.logvar_layer = nn.ModuleList()

        for _ in range(self.n_concepts):
            self.mu_layer.append(nn.Linear(self.in_size, self.emb_size))  
            self.logvar_layer.append(nn.Linear(self.in_size, self.emb_size))

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
        if not len(c_pred.shape)>2:
            c_pred = c_pred.unsqueeze(-1)
        cloned_c_pred = c_pred.detach().clone().expand(-1, self.prototype_emb_pos.shape[-1],-1)
        cloned_c_pred = torch.where(cloned_c_pred>0.5,1,0)
        prototype_emb = cloned_c_pred * self.prototype_emb_pos[None, concept_idx, :, None] + \
                        (1 - cloned_c_pred) * self.prototype_emb_neg[None, concept_idx, :, None]
        if len(c_emb.shape)>2:
            c_int = c_int.unsqueeze(2).expand(-1, -1, self.prototype_emb_pos.shape[-1], -1).squeeze()
            #prototype_emb = prototype_emb.unsqueeze(-1).expand(-1,-1,c_emb.shape[-1])
        else:
            c_int = c_int.unsqueeze(1).expand(-1, self.prototype_emb_pos.shape[-1], -1).int()
            c_emb = c_emb.unsqueeze(-1)
        int_emb = c_int * prototype_emb + (1 - c_int) * c_emb
        return int_emb.squeeze()

    def reparameterize(self, mu, logvar):
        expanded_logvar = logvar.unsqueeze(-1).expand(-1,-1,self.mc_samples)
        expanded_mu = mu.unsqueeze(-1).expand(-1,-1,self.mc_samples)
        std = torch.exp(0.5 * expanded_logvar)
        eps = torch.randn_like(std)
        return expanded_mu + eps * std

    def forward(self, x, c, noise=None, p_int=None):
        bsz = x.shape[0]
        p_int = self.p_int_train if self.training else p_int
        if self.train_backbone:
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
        if noise!=None:
            eps = torch.randn_like(x)
            x = eps * noise + x * (1-noise)
        c_pred_list, c_emb_list, mu_list, logvar_list = [], [], [], []
        x = self.shared_layers(x)
        for i in range(self.n_concepts):
            mu = self.mu_layer[i](x)
            logvar = self.logvar_layer[i](x)
            if self.training:
                c_emb = self.reparameterize(mu, logvar) 
                distance = torch.norm(c_emb - self.prototype_emb_neg[i,:][None,:,None], dim=1) \
                            - torch.norm(c_emb - self.prototype_emb_pos[i,:][None,:,None], dim=1)
                distance = torch.sigmoid(self.a * distance)
                distance = torch.sum(distance, axis=-1)
                c_pred = distance / self.mc_samples                   
            else:
                c_emb = mu
                distance = torch.norm(c_emb - self.prototype_emb_neg[i,:], dim=1) \
                            - torch.norm(c_emb - self.prototype_emb_pos[i,:], dim=1)
                c_pred = torch.sigmoid(self.a * distance)
            if p_int!=None:
                repeat_param = self.mc_samples if self.training else None
                int_mask, c_inter = get_intervened_concepts_predictions(c_pred.unsqueeze(-1), 
                                                                       c[:,i].unsqueeze(-1), 
                                                                       p_int, 
                                                                       True, 
                                                                       self.training,
                                                                       repeat_param)
                c_emb = self.apply_intervention(c_inter, int_mask, c_emb, i)
            c_emb_list.append(c_emb.unsqueeze(1))
            c_pred_list.append(c_pred.unsqueeze(1))
            mu_list.append(mu.unsqueeze(1))
            logvar_list.append(logvar.unsqueeze(1))

        c_emb = torch.cat(c_emb_list, dim=1) 
        c_pred = torch.cat(c_pred_list, dim=1) 
        mu = torch.cat(mu_list, dim=1) 
        logvar = torch.cat(logvar_list, dim=1)
        y_pred = self.classify(c_emb)
        return c_pred, y_pred, c_emb, mu, logvar

    def classify(self, c_emb):
        bsz = c_emb.shape[0]
        if self.training:
            probs = 0
            for i in range(self.mc_samples):
                weights = self.fc(c_emb[:,:,:,i].flatten(start_dim=1))
                logits = -self.d * torch.norm(weights.unsqueeze(1).expand(-1, self.n_classes, -1)-self.task_emb.unsqueeze(0).expand(bsz, -1, -1), dim=-1)
                probs += nn.functional.softmax(logits, dim=-1)
            probs /= self.mc_samples
        else:
            weights = self.fc(c_emb.flatten(start_dim=1))
            logits = -self.d * torch.norm(weights.unsqueeze(1).expand(-1, self.n_classes, -1)-self.task_emb.unsqueeze(0).expand(bsz, -1, -1), dim=-1)
            probs = nn.functional.softmax(logits, dim=-1)
        return probs

    def D_kl_gaussian(self, mu_q, logvar_q):
        # the D_kl divergence between two Gaussian distributions, the prior is a standard gaussian
        D_kl = 0.5 * (logvar_q.exp() + (mu_q**2) - 1 - logvar_q).sum(dim=-1)
        return D_kl.mean()

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
            concept_loss += concept_form(c_pred[:,i].squeeze(), c[:,i])
        concept_loss /= self.n_concepts 
        # task loss
        task_loss = task_form(y_pred, y.long())
        # KL divergence
        D_kl = self.D_kl_gaussian(mu.flatten(start_dim=1), logvar.flatten(start_dim=1))
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
