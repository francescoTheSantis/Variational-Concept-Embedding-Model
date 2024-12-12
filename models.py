import torch
import torch.nn as nn
import torch.nn.functional as F

class AA_CEM(nn.Module):
    def __init__(self, embedding_size, n_concepts, concept_size, classifier, n_labels, p_int=0.1):
        super(AA_CEM, self).__init__()
        self.embedding_size = embedding_size
        self.n_concepts = n_concepts
        self.concept_size = concept_size
        self.classifier = classifier
        self.n_labels = n_labels
        self.p_int = p_int

        self.concept_embedding_layers = nn.ModuleList()
        for i in range(self.n_concepts):
            layers = nn.Sequential(
                nn.Linear(self.embedding_size, self.concept_size),
                nn.ReLU(),
                nn.Linear(self.concept_size, self.concept_size),
                nn.LeakyReLU(0.1)
            )
            self.concept_embedding_layers.append(layers)
        
        self.concept_scorers = nn.ModuleList()
        for i in range(self.n_concepts):
            layers = nn.Sequential(
                nn.Linear(self.concept_states[i] * self.concept_size, self.concept_states[i] * self.concept_size),
                nn.ReLU(),
                nn.Linear(self.concept_states[i] * self.concept_size, self.concept_states[i]),
                nn.Softmax(dim=1)
            )
            self.concept_scorers.append(layers)

        if self.classifier == 'linear':
            self.weights_generator = torch.nn.ModuleList()
            for i in range(self.n_labels):
                self.weights_generator.append(
                    nn.Sequential(
                        nn.Linear(self.concept_size, self.concept_size), 
                        nn.ReLU(), 
                        nn.Linear(self.concept_size, 1)
                    )
                )  
        elif self.classifier == 'cem':
            self.mlp = nn.Sequential(
                nn.Linear(self.n_concepts * self.concept_size, self.n_concepts * self.concept_size),
                nn.ReLU(),
                nn.Linear(self.n_concepts * self.concept_size, self.n_labels)
            )
        else:
            raise ValueError('Invalid classifier type')

    def apply_intervention(self, c_pred, c_int, device):
        mask = torch.bernoulli(torch.full(c_pred.shape, self.p_int)).to(device)
        c_pred = c_pred * (1 - mask) + c_int * mask
        return c_pred
    
    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std
    
    def forward(self, sentence_batch, c_int, device='cuda'):
        x = self.encoder(sentence_batch['input_ids'].squeeze().to(torch.long).to(device), 
                         sentence_batch['attention_mask'].squeeze().to(torch.long).to(device), 
                         output_hidden_states=True).hidden_states[-1][:,0,:]
        
        bsz = x.shape[0]

        c_pred_list, c_emb_list = [], []
        c_int  = c_int.view(-1, self.n_concepts, self.concept_states[0]) # (batch_size, n_concepts, n_states)
        for i in range(self.n_concepts):
            c_emb = self.concept_embedding_layers[i](x) # (batch_size, n_states * emb_size)
            c_pred = self.concept_scorers[i](c_emb) # (batch_size, n_states)
            # apply intervention
            c_pred = self.apply_intervention(c_pred, c_int[:,i,:], device)
            c_emb = c_emb.view(-1, self.concept_states[i], self.concept_size) # (batch_size, n_states, emb_size)
            c_emb = torch.bmm(c_pred.unsqueeze(1), c_emb) # (batch_size, emb_size)
            c_emb_list.append(c_emb)
            c_pred_list.append(c_pred.unsqueeze(1))
        
        c_emb = torch.cat(c_emb_list, dim=1) # (batch_size, n_concepts, emb_size)
        c_pred = torch.cat(c_pred_list, dim=1) # (batch_size, n_concepts, n_states)
        
        if self.classifier == 'linear':
            y = torch.zeros(bsz, self.n_labels).to(device)
            logits = torch.zeros(bsz, self.n_concepts, self.n_labels).to(device)
            for i in range(self.n_labels):
                weights = self.weights_generator[i](c_emb) # batch, n_concepts, 1
                y[:,i] = torch.bmm(c_pred.max(-1).values.unsqueeze(1), weights).squeeze() 
                logits[:, :, i] = weights.squeeze() * c_pred.max(-1).values    
            return y, logits, c_emb, c_pred
        elif self.classifier == 'cem':
            logits = torch.zeros(bsz, self.n_concepts, self.n_labels).to(device)
            y = self.mlp(c_emb.view(bsz, -1))
            return y, logits, c_emb, c_pred
        else:   
            raise ValueError('Invalid classifier type')

             

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