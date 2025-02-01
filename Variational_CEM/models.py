import torch
import torch.nn as nn
import pytorch_lightning as pl
from utilities import get_intervened_concepts_predictions, D_kl_gaussian

class V_CEM(pl.LightningModule):
    def __init__(self, 
                 in_size, 
                 n_concepts, 
                 emb_size, 
                 embedding_interventions=True, 
                 sampling=False):
        super().__init__()

        self.in_size = in_size
        self.n_concepts = n_concepts
        self.emb_size = emb_size
        self.embedding_interventions = embedding_interventions
        self.sampling = sampling

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
    
    def forward(self, x, c=None, p_int=0):
        bsz = x.shape[0]
        device = x.device
        c_pred_list, c_emb_list, mu_list, logvar_list = [], [], [], []
        x = self.shared_layers(x)
        for i in range(self.n_concepts):
            c_pred = self.concept_scorers[i](x) 
            if c!=None and p_int>0 and self.embedding_interventions==False:
                c_pred = get_intervened_concepts_predictions(c_pred, c[:,i].unsqueeze(-1), p_int, False)
            emb = self.layers[i](torch.cat([x, c_pred], dim=-1))
            mu = self.mu_layer[i](emb)
            logvar = self.logvar_layer[i](emb)
            # during training we sample from the multivariate normal distribution, at test-time we take MAP.
            if self.training and self.sampling:
                c_emb = self.reparameterize(mu, logvar)
            else:
                c_emb = mu
            # apply prototype interventions
            if c!=None and p_int>0 and self.embedding_interventions:
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
        return c_pred, c_emb, mu, logvar

    def training_step(self, batch, batch_idx):
        D_kl = 0
        concept_loss = 0
    


if __name__ == '__main__':
    # Test model
    model = V_CEM(10, 5, 3)
    x = torch.randn(5, 10)
    c = torch.randint(0, 2, (5, 5))
    p_int = 0.5
    c_pred, c_emb, mu, logvar = model(x, c, p_int)
    print(c_pred.shape, c_emb.shape, mu.shape, logvar.shape)