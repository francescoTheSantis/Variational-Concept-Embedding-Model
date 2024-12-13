import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utilities import D_kl_gaussian
    
@torch.no_grad()
def evaluate(model, concept_encoder, classifier, loaded_set, n_labels, n_concepts, emb_size,
             concept_form=None, task_form=None, device='cuda'):
    if concept_encoder!=None:
        concept_encoder.eval()
    classifier.eval()
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
        if model=='e2e':
            y_pred = classifier(concept_encoder(batch[0].to(device)))
        elif model=='cem':
            c_pred, c_emb = concept_encoder(batch[0].to(device), None, concept_labels, train)
            y_pred = classifier(c_emb)
        elif model=='aa_cem':
            c_pred, c_emb, mu, logvar = concept_encoder(batch[0].to(device), concept_labels)
            y_pred = classifier(c_emb)
        elif 'cbm' in model:
            c_pred = concept_encoder(batch[0].to(device))
            y_pred = classifier(c_pred)

        D_kl = 0
        concept_loss = 0
        if 'cem' in model or 'cbm' in model:
            for i in range(n_concepts):
                concept_loss += concept_form(c_pred[:,i], concept_labels[:,i])
            concept_loss /= n_concepts  

        y_pred = y_pred.squeeze()          
        task_loss = task_form(y_pred, y)  
        running_task_loss += task_loss.item()
        running_concept_loss += concept_loss.item()
        if model=='aa_cem':
            prototype_emb_pos = concept_encoder.prototype_emb_pos
            prototype_emb_neg = concept_encoder.prototype_emb_neg
            cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, prototype_emb_pos.shape[-1])
            prototype_emb_pos = prototype_emb_pos.unsqueeze(0).expand(c_pred.size(0), -1, -1)
            prototype_emb_neg = prototype_emb_neg.unsqueeze(0).expand(c_pred.size(0), -1, -1)
            prototype_emb = cloned_c_pred * prototype_emb_pos + (1 - cloned_c_pred) * prototype_emb_neg
            D_kl = D_kl_gaussian(mu, logvar, prototype_emb)
            running_d_kl_loss += D_kl.item()

        concept_preds = torch.cat([concept_preds, c_pred.argmax(-1)])
        running_task_loss += task_loss.item()
        running_concept_loss += concept_loss.item()  
        y_pred = torch.where(y_pred>0,1,0) #y_pred.argmax(-1)
        true_concepts = torch.cat([true_concepts, concept_labels])
        c_embs = torch.cat([c_embs, c_emb])
        task_preds = torch.cat([task_preds, y_pred])
        real_labels = torch.cat([real_labels, y])
    if concept_encoder!=None:
        concept_encoder.train()
    classifier.train()
    
    return running_task_loss/len(loaded_set), running_concept_loss/len(loaded_set), running_d_kl_loss/len(loaded_set), \
task_preds[1:], real_labels[1:] , concept_preds[1:,:], true_concepts[1:,:], c_embs[1:,:,:]



def train(model, loaded_train, loaded_val, loaded_test, concept_encoder, classifier, lr, epochs, n_labels, 
          n_concepts, emb_size, step_size, gamma, device='cuda'):
    
    if model=='e2e':
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
        print('Number of trainable parameters:', sum(p.numel() if p.requires_grad==True else 0 for p in classifier.parameters()))
    else:
        optimizer = torch.optim.AdamW(nn.Sequential(concept_encoder, classifier).parameters(), lr=lr)
        print('Number of trainable parameters:', sum(p.numel() if p.requires_grad==True else 0 for p in concept_encoder.parameters())+\
          sum(p.numel() if p.requires_grad==True else 0 for p in classifier.parameters()))

    concept_form = nn.BCELoss()
    task_form = nn.BCEWithLogitsLoss() # so far we used only binary classification datasets 
    train_task_losses = []
    train_concept_losses = []
    D_kl_losses = []
    val_D_kl_losses = []
    val_task_losses = []
    val_concept_losses = []
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    concept_encoder.train()
    classifier.train()
    for epoch in range(epochs):
        running_task_loss = 0
        running_concept_loss = 0
        running_d_kl_loss = 0
        for batch in tqdm(loaded_train):
            optimizer.zero_grad()
            y = torch.Tensor(batch[2]).to(device)
            concept_labels = batch[1].to(device)
            if model=='e2e':
                y_pred = classifier(concept_encoder(batch[0].to(device)))
            elif model=='cem':
                c_pred, c_emb = concept_encoder(batch[0].to(device), None, concept_labels, train)
                y_pred = classifier(c_emb)
            elif model=='aa_cem':
                c_pred, c_emb, mu, logvar = concept_encoder(batch[0].to(device), concept_labels)
                y_pred = classifier(c_emb)
            elif 'cbm' in model:
                c_pred = concept_encoder(batch[0].to(device))
                y_pred = classifier(c_pred)

            D_kl = 0
            concept_loss = 0
            if 'cem' in model or 'cbm' in model:
                for i in range(n_concepts):
                    concept_loss += concept_form(c_pred[:,i], concept_labels[:,i])
                concept_loss /= n_concepts  

            y_pred = y_pred.squeeze()          
            task_loss = task_form(y_pred, y)  
            running_task_loss += task_loss.item()
            running_concept_loss += concept_loss.item()
            if model=='e2e':
                loss = task_loss
            elif model=='cem' or 'cbm' in model:
                loss = concept_loss + task_loss
            elif model=='aa_cem':
                prototype_emb_pos = concept_encoder.prototype_emb_pos
                prototype_emb_neg = concept_encoder.prototype_emb_neg
                cloned_c_pred = c_pred.detach().clone().unsqueeze(-1).expand(-1, -1, prototype_emb_pos.shape[-1])
                prototype_emb_pos = prototype_emb_pos.unsqueeze(0).expand(c_pred.size(0), -1, -1)
                prototype_emb_neg = prototype_emb_neg.unsqueeze(0).expand(c_pred.size(0), -1, -1)
                prototype_emb = cloned_c_pred * prototype_emb_pos + (1 - cloned_c_pred) * prototype_emb_neg
                D_kl = D_kl_gaussian(mu, logvar, prototype_emb)
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
        'n_labels':n_labels,
        'n_concepts':n_concepts, 
        'device':device,
        'concept_form':concept_form,
        'task_form':task_form,
        'test':False,
        'emb_size': emb_size 
        }
        
        val_task_loss, val_concept_loss, val_d_kl_loss, _, _, _, _ = evaluate(**params)
        val_task_losses.append(val_task_loss)
        val_concept_losses.append(val_concept_loss)
        val_D_kl_losses.append(val_d_kl_loss)

    params['loaded_set'] = loaded_test
    params['test'] = True
    y_preds, y, c_preds, c_true, c_emb = evaluate(**params)

    return concept_encoder, classifier, train_task_losses, train_concept_losses, D_kl_losses, val_task_losses, val_concept_losses, \
        val_D_kl_losses, y_preds, y, c_preds, c_true, c_emb

