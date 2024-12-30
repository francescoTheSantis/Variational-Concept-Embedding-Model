import torch
from torch import nn
from torch.nn import functional as f
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utilities import D_kl_gaussian, get_intervened_concepts_predictions
    
kl_penalty = 1e-1

@torch.no_grad()
def evaluate(model, concept_encoder, classifier, loaded_set, n_concepts, emb_size,
             concept_form=None, task_form=None, device='cuda', corruption=0, n_labels=None, intervention_prob=0):
    if concept_encoder!=None:
        concept_encoder.eval()
        concept_encoder.to(device)
    classifier.eval()
    classifier.to(device)
    running_task_loss = 0
    running_concept_loss = 0
    running_d_kl_loss = 0
    task_preds = torch.zeros(1).to(device)
    concept_preds =  torch.zeros(1,n_concepts).to(device)
    true_concepts =  torch.zeros(1,n_concepts).to(device)
    c_embs = torch.zeros(1, n_concepts, emb_size).to(device)        
    real_labels = torch.zeros(1).to(device)
    # to avoid errors when using e2e or cbm
    c_emb = torch.zeros(1, n_concepts, emb_size).to(device)
    c_pred = torch.zeros(1, n_concepts).to(device)
    for batch in loaded_set:
        y = torch.Tensor(batch[2]).to(device)
        concept_labels = batch[1].to(device)
        x = batch[0].to(device)
        if corruption>0:
            eps = torch.randn_like(x).to(device)
            x = eps * corruption + x * (1-corruption)

        if model=='e2e':
            y_pred = classifier(x)
        elif model=='cem':
            c_emb, c_pred = concept_encoder(x, None, concept_labels, intervention_prob, False)
            y_pred = classifier(c_emb.flatten(start_dim=1))
        elif model=='aa_cem':
            c_pred, c_emb, mu, logvar = concept_encoder(x, concept_labels, intervention_prob)
            y_pred = classifier(c_emb.flatten(start_dim=1))
        elif 'cbm' in model:
            c_pred = concept_encoder(x)
            if intervention_prob>0:
                c_pred = get_intervened_concepts_predictions(c_pred, concept_labels, intervention_prob)
            y_pred = classifier(c_pred)

        D_kl = 0
        concept_loss = 0
        if 'cem' in model or 'cbm' in model:
            for i in range(n_concepts):
                concept_loss += concept_form(c_pred[:,i], concept_labels[:,i])
            concept_loss /= n_concepts  

        y_pred = y_pred.squeeze()
        y = y.squeeze().long() 

        task_loss = task_form(y_pred, y) # .long()   

        running_task_loss += task_loss.item()
        if concept_encoder!=None:
            running_concept_loss += concept_loss.item()
        if model=='aa_cem':
            '''
            prototype_emb_pos = concept_encoder.prototype_emb_pos
            prototype_emb_neg = concept_encoder.prototype_emb_neg
            cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, prototype_emb_pos.shape[-1])
            prototype_emb_pos = prototype_emb_pos.unsqueeze(0).expand(c_pred.size(0), -1, -1)
            prototype_emb_neg = prototype_emb_neg.unsqueeze(0).expand(c_pred.size(0), -1, -1)
            prototype_emb = cloned_c_pred * prototype_emb_pos + (1 - cloned_c_pred) * prototype_emb_neg
            '''
            cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, concept_encoder.emb_size)
            prototype_emb = cloned_c_pred * concept_encoder.prototype_emb_pos[None, :, :] + \
                (1 - cloned_c_pred) * concept_encoder.prototype_emb_neg[None, :, :]
            D_kl = D_kl_gaussian(mu, logvar, prototype_emb) * kl_penalty
            running_d_kl_loss += D_kl.item()

        concept_preds = torch.cat([concept_preds, c_pred])
        y_pred = y_pred.argmax(-1) 
        # y_pred = torch.where(y_pred>0,1,0) 
        true_concepts = torch.cat([true_concepts, concept_labels])
        #print(c_embs.shape, c_emb.shape)
        c_embs = torch.cat([c_embs, c_emb])
        task_preds = torch.cat([task_preds, y_pred])
        real_labels = torch.cat([real_labels, y])
    if concept_encoder!=None:
        concept_encoder.train()
    classifier.train()
    
    return running_task_loss/len(loaded_set), running_concept_loss/len(loaded_set), running_d_kl_loss/len(loaded_set), \
task_preds[1:], real_labels[1:] , concept_preds[1:,:], true_concepts[1:,:], c_embs[1:,:,:]



def train(model, loaded_train, loaded_val, loaded_test, concept_encoder, classifier, lr, epochs, 
          n_concepts, emb_size, step_size, gamma, test, n_labels, corruption=0, device='cuda'):
    
    concept_form = nn.BCELoss()
    task_form = nn.CrossEntropyLoss() 
    #task_form = nn.BCEWithLogitsLoss() # so far we used only binary classification datasets 
    train_task_losses = []
    train_concept_losses = []
    D_kl_losses = []
    val_D_kl_losses = []
    val_task_losses = []
    val_concept_losses = []
    if concept_encoder!=None:
        concept_encoder.train()
        concept_encoder.to(device)
    classifier.train()
    classifier.to(device)

    if test:
        params = {
            'model': model,
            'loaded_set':loaded_test,
            'concept_encoder':concept_encoder, 
            'classifier':classifier,
            'n_concepts':n_concepts, 
            'device':device,
            'concept_form':concept_form,
            'task_form':task_form,
            'emb_size': emb_size,
            'corruption': corruption,
            'n_labels': n_labels
        }
        _, _, _, y_preds, y, c_preds, c_true, c_emb = evaluate(**params)
        return y_preds, y, c_preds, c_true, c_emb

    if model=='e2e':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        print('Number of trainable parameters:', sum(p.numel() if p.requires_grad==True else 0 for p in classifier.parameters()))
    else:
        optimizer = torch.optim.AdamW(nn.Sequential(concept_encoder, classifier).parameters(), lr=lr)
        print('Number of trainable parameters:', sum(p.numel() if p.requires_grad==True else 0 for p in concept_encoder.parameters())+\
          sum(p.numel() if p.requires_grad==True else 0 for p in classifier.parameters()))
        
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for _ in range(epochs):
        running_task_loss = 0
        running_concept_loss = 0
        running_d_kl_loss = 0
        for batch in tqdm(loaded_train):
            optimizer.zero_grad()
            y = torch.Tensor(batch[2]).to(device)
            concept_labels = batch[1].to(device)
            x = batch[0].to(device)
            if model=='e2e':
                y_pred = classifier(x)
            elif model=='cem':
                c_emb, c_pred = concept_encoder(x, None, concept_labels, True)
                y_pred = classifier(c_emb.flatten(start_dim=1))
            elif model=='aa_cem':
                c_pred, c_emb, mu, logvar = concept_encoder(x, concept_labels)
                y_pred = classifier(c_emb.flatten(start_dim=1))
            elif 'cbm' in model:
                c_pred = concept_encoder(x)
                y_pred = classifier(c_pred)

            D_kl = 0
            concept_loss = 0
            if 'cem' in model or 'cbm' in model:
                for i in range(n_concepts):
                    concept_loss += concept_form(c_pred[:,i], concept_labels[:,i])
                concept_loss /= n_concepts  

            y_pred = y_pred.squeeze()
            y = y.squeeze().long() 
            
            task_loss = task_form(y_pred, y) # .long()  
            running_task_loss += task_loss.item()
            if concept_encoder!=None:
                running_concept_loss += concept_loss.item()
            if model=='e2e':
                loss = task_loss
            elif model=='cem' or 'cbm' in model:
                loss = concept_loss + task_loss
            elif model=='aa_cem':
                '''
                prototype_emb_pos = concept_encoder.prototype_emb_pos
                prototype_emb_neg = concept_encoder.prototype_emb_neg
                cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, prototype_emb_pos.shape[-1])
                prototype_emb_pos = prototype_emb_pos.unsqueeze(0).expand(c_pred.size(0), -1, -1)
                prototype_emb_neg = prototype_emb_neg.unsqueeze(0).expand(c_pred.size(0), -1, -1)
                prototype_emb = cloned_c_pred * prototype_emb_pos + (1 - cloned_c_pred) * prototype_emb_neg
                '''
                cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, concept_encoder.emb_size)
                prototype_emb = cloned_c_pred * concept_encoder.prototype_emb_pos[None, :, :] + \
                    (1 - cloned_c_pred) * concept_encoder.prototype_emb_neg[None, :, :]
                D_kl = D_kl_gaussian(mu, logvar, prototype_emb) * kl_penalty
                running_d_kl_loss += D_kl.item()
                loss = concept_loss + task_loss + D_kl
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_task_losses.append(running_task_loss/len(loaded_train))
        train_concept_losses.append(running_concept_loss/len(loaded_train))  
        D_kl_losses.append(running_d_kl_loss/len(loaded_train))
        
        params = {
            'model': model,
            'loaded_set':loaded_val,
            'concept_encoder':concept_encoder, 
            'classifier':classifier,
            'n_concepts':n_concepts, 
            'device':device,
            'concept_form':concept_form,
            'task_form':task_form,
            'emb_size': emb_size,
            'n_labels': n_labels
        }
        
        val_task_loss, val_concept_loss, val_d_kl_loss, _, _, _, _, _ = evaluate(**params)
        val_task_losses.append(val_task_loss)
        val_concept_losses.append(val_concept_loss)
        val_D_kl_losses.append(val_d_kl_loss)

    params['loaded_set'] = loaded_test
    _, _, _, y_preds, y, c_preds, c_true, c_emb = evaluate(**params)

    return concept_encoder, classifier, train_task_losses, train_concept_losses, D_kl_losses, val_task_losses, val_concept_losses, \
        val_D_kl_losses, y_preds, y, c_preds, c_true, c_emb
