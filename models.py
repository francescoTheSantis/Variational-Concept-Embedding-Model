import torch
import torch.nn as nn
import torch.nn.functional as F

class AA_CEM(nn.Module):
    def __init__(self, in_size, n_concepts, emb_size, p_int=0):
        super(AA_CEM, self).__init__()
        self.in_size = in_size
        self.n_concepts = n_concepts
        self.emb_size = emb_size

        # Initialize learnable prototypes
        self.prototype_emb_pos = nn.Parameter(torch.randn(n_concepts, emb_size))
        self.prototype_emb_neg = nn.Parameter(torch.randn(n_concepts, emb_size))

        self.concept_scorers = nn.ModuleList()
        for _ in range(self.n_concepts):
            layers = nn.Sequential(
                nn.Linear(self.in_size, self.in_size),
                nn.ReLU(),
                nn.Linear(self.in_size, 1),
                nn.Sigmoid()
            )
            self.concept_scorers.append(layers)

        self.layers = nn.ModuleList()
        for _ in range(self.n_concepts):
            layers = nn.Sequential(
                nn.Linear(self.in_size + 1, self.in_size + 1),
                nn.ReLU()
            )
            self.layers.append(layers)

        self.mu_layer = nn.Linear(self.in_size + 1, self.emb_size)      
        self.logvar_layer = nn.Linear(self.in_size + 1, self.emb_size)  
        
    def apply_intervention(self, c_pred, c_int, c_emb, device):
        cloned_c_pred = c_pred.detach().clone()
        mask = torch.bernoulli(torch.full(cloned_c_pred.shape, self.p_int)).to(device)
        cloned_c_pred = mask * c_int + cloned_c_pred * (1 - mask)
        cloned_c_pred = cloned_c_pred.unsqueeze(-1).expand(-1, -1, prototype_emb_pos.shape[-1])
        prototype_emb_pos = prototype_emb_pos.unsqueeze(0).expand(c_pred.size(0), -1, -1)
        prototype_emb_neg = prototype_emb_neg.unsqueeze(0).expand(c_pred.size(0), -1, -1)
        prototype_emb = cloned_c_pred * prototype_emb_pos + (1 - cloned_c_pred) * prototype_emb_neg
        c_emb = mask * prototype_emb + (1 - mask) * c_emb
        return c_pred
    

    def reparameterize(self, mu, logvar):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * logvar)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std
    
    def forward(self, x, c_int=None, device='cuda'):
        bsz = x.shape[0]
        c_pred_list, c_emb_list, mu_list, logvar_list = [], [], [], []
        for i in range(self.n_concepts):
            c_pred = self.concept_scorers[i](x) 
            # apply intervention
            #c_pred = self.apply_intervention(c_pred, c_int[:,i,:], device)
            emb = self.layers(torch.cat([x, c_pred.unsqueeze(-1)], dim=-1))
            mu = self.mu_layer(emb)
            logvar = self.logvar_layer(emb)
            # during training we sample from the multivariate normal distribution, at test-time we take MAP.
            if self.training:
                c_emb = self.reparameterize(mu, logvar)
            else:
                c_emb = mu
            # apply prototype interventions
            c_emb = self.apply_intervention(c_pred, c_int, c_emb, device)
            c_emb_list.append(c_emb)
            c_pred_list.append(c_pred.unsqueeze(1))
            mu_list.append(mu.unsqueeze(1))
            logvar_list.append(logvar.unsqueeze(1))
        
        c_emb = torch.cat(c_emb_list, dim=1) # (batch_size, n_concepts, emb_size)
        c_pred = torch.cat(c_pred_list, dim=1) # (batch_size, n_concepts, n_states)
        mu = torch.cat(mu_list, dim=1)
        logvar = torch.cat(logvar_list, dim=1)
        
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

    def forward(self, x, intervention_idxs=None, c=None, train=False):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            c_pred_list.append(c_pred)
            # Time to check for interventions
            c_pred = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=used_int_idxs,
                c_true=c,
                train=train,
            )

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))

        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)