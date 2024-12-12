import torch
import torch.nn.functional as F
from torch import nn
#from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from models import ConceptEmbedding, AA_CEM
#from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    
@torch.no_grad()
def evaluate_cem(loaded_set, m_lem, n_labels, n_concepts, concept_states, 
                 max_length, concept_names, concept_form=None, task_form=None, explain=False, device='cuda'):
    m_lem.eval()
    running_task_loss = 0
    running_concept_loss = 0
    task_preds = torch.zeros(1).to(device)
    concept_preds =  torch.zeros(1,n_concepts).to(device)
    true_concepts =  torch.zeros(1,n_concepts).to(device)
    pred_logits = torch.zeros(1, n_concepts, n_labels).to(device)        
    original_sentences = torch.zeros(1,max_length).to(device)
    real_labels = torch.zeros(1).to(device)
    concept_probs = torch.zeros(1, n_concepts, concept_states[0]).to(device)
    for sentence_batch in loaded_set:
        y = torch.Tensor(sentence_batch['labels']).to(device)#.to(torch.long)
        original_sentences = torch.cat([original_sentences, sentence_batch['input_ids'].to(device)])
        concept_labels = torch.cat([torch.nn.functional.one_hot(sentence_batch[f'{name}'].to(torch.long), concept_states[idx]).to(torch.float) for idx, name in enumerate(concept_names)], axis=1).to(device)
        true_concepts = torch.cat([true_concepts, concept_labels.view(-1, n_concepts, concept_states[0]).argmax(-1)])
        y_pred, logits, _, c_pred = m_lem(sentence_batch, concept_labels)
        y_pred = y_pred.squeeze()
        concept_loss = 0
        for i in range(n_concepts):
            concept_loss += concept_form(c_pred[:,i,:], concept_labels.view(-1, n_concepts, concept_states[i])[:,i,:])
        concept_loss /= n_concepts            
        task_loss = task_form(y_pred, y)  
        concept_preds = torch.cat([concept_preds, c_pred.argmax(-1)])
        concept_probs = torch.cat([concept_probs, c_pred])
        pred_logits = torch.cat([pred_logits, logits])
        running_task_loss += task_loss.item()
        running_concept_loss += concept_loss.item()  
        y_pred = torch.where(y_pred>0,1,0) #y_pred.argmax(-1)
        task_preds = torch.cat([task_preds, y_pred])
        real_labels = torch.cat([real_labels, y])
    m_lem.train()
    
    if explain:
        return task_preds[1:], real_labels[1:] , concept_preds[1:,:], \
               true_concepts[1:,:], pred_logits[1:,:,:], original_sentences[1:,:], concept_probs[1:,:,:]
    else:
        return running_task_loss/len(loaded_set), running_concept_loss/len(loaded_set), \
               task_preds[1:], real_labels[1:] , concept_preds[1:,:], true_concepts[1:,:]



def train(model, loaded_train, loaded_val, loaded_test, concept_encoder, classifier, lr, epochs, n_labels, 
          n_concepts, step_size, gamma, lambda_coeff=1, device='cuda'):
    
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
    val_task_losses = []
    val_concept_losses = []
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    concept_encoder.train()
    classifier.train()
    for epoch in range(epochs):
        running_task_loss = 0
        running_concept_loss = 0
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
                c_pred, c_emb = concept_encoder(batch[0].to(device), concept_labels)
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
                loss = concept_loss + lambda_coeff * task_loss
            elif model=='aa_cem':
                D_kl
                loss = concept_loss + lambda_coeff * task_loss
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_task_losses.append(running_task_loss/len(loaded_train))
        train_concept_losses.append(running_concept_loss/len(loaded_train))   
        
        params = {
        'loaded_set':loaded_val,
        'm_lem':m_lem, 
        'n_labels':n_labels,
        'n_concepts':n_concepts, 
        'concept_states':concept_states, 
        'max_length':max_length, 
        'explain':False, 
        'device':device,
        'concept_form':concept_form,
        'task_form':task_form,
        'concept_names': concept_names,
        'explain': False
        }
        
        val_task_loss, val_concept_loss, _, _, _, _ = evaluate_icem(**params)
        val_task_losses.append(val_task_loss)
        val_concept_losses.append(val_concept_loss)

    params['loaded_set'] = loaded_test
    params['explain'] = True
    y_preds, y, c_preds, c_true, logits, original_sentences, concept_probs = evaluate_icem(**params)

    return m_lem, train_task_losses, train_concept_losses, val_task_losses, val_concept_losses, y_preds, y, c_preds, c_true, logits, original_sentences, concept_probs

