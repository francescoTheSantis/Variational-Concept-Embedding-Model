import torch
import torch.nn as nn
from utilities import get_intervened_concepts_predictions

class AA_CEM(nn.Module):
    def __init__(self, in_size, n_concepts, emb_size, embedding_interventions=True, sampling=False):
        super(AA_CEM, self).__init__()
        self.in_size = in_size
        self.n_concepts = n_concepts
        self.emb_size = emb_size
        self.embedding_interventions = embedding_interventions
        self.sampling = sampling

        # Initialize learnable concept prototypes
        self.prototype_emb_pos = nn.Parameter(torch.randn(n_concepts, emb_size))
        self.prototype_emb_neg = nn.Parameter(torch.randn(n_concepts, emb_size))

        self.concept_scorers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.mu_layer = nn.ModuleList()
        self.logvar_layer = nn.ModuleList()
        for _ in range(self.n_concepts):
            layers = nn.Sequential(
                nn.Linear(self.in_size, self.in_size),
                nn.ReLU(),
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
    
    def forward(self, x, c=None, p_int=0, device='cuda'):
        #bsz = x.shape[0]
        c_pred_list, c_emb_list, mu_list, logvar_list = [], [], [], []
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
        
        c_emb = torch.cat(c_emb_list, dim=1) # (batch_size, n_concepts, emb_size)
        c_pred = torch.cat(c_pred_list, dim=1)[:,:,0] # (batch_size, n_concepts)
        mu = torch.cat(mu_list, dim=1) # (batch_size, n_concepts, emb_size)
        logvar = torch.cat(logvar_list, dim=1) # (batch_size, n_concepts, emb_size)
        
        return c_pred, c_emb, mu, logvar

             


class ConceptEmbedding(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
            torch.nn.Sigmoid(),
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, intervention_idxs=None, c=None, p_int=0, train=False):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            c_pred_list.append(c_pred)
            '''
            # Time to check for interventions
            c_pred = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=used_int_idxs,
                c_true=c,
                train=train,
            )
            '''
            if c!=None and p_int>0:
                c_pred = get_intervened_concepts_predictions(c_pred, c[:,i].unsqueeze(-1), p_int)

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))

        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)