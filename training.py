import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from models import MultiStateModel
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AutoTokenizer
from openai import OpenAI
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    
# ------------------------------------------------- AA-CEM -------------------------------------------------

@torch.no_grad()
def evaluate_icem(loaded_set, m_lem, n_labels, n_concepts, concept_states, 
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


def train_icem(loaded_train, loaded_val, loaded_test, m_lem, max_length, concept_names, lr, epochs, n_labels, 
               n_concepts, concept_states, step_size, gamma, lambda_coeff, device='cuda'):
    print('Number of trainable parameters:', sum(p.numel() if p.requires_grad==True else 0 for p in m_lem.parameters()))
    optimizer = torch.optim.AdamW(m_lem.parameters(), lr=lr)
    concept_form =  nn.BCELoss()
    # since both the datasets are have binary labels, we can use BCELoss
    task_form = nn.BCEWithLogitsLoss()
    train_task_losses = []
    train_concept_losses = []
    val_task_losses = []
    val_concept_losses = []
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    m_lem.train()
    for epoch in range(epochs):
        running_task_loss = 0
        running_concept_loss = 0
        for sentence_batch in tqdm(loaded_train):
            optimizer.zero_grad()
            y = torch.Tensor(sentence_batch['labels'])
            concept_labels = torch.cat([torch.nn.functional.one_hot(sentence_batch[f'{name}'].to(torch.long), concept_states[idx]).to(torch.float) for idx, name in enumerate(concept_names)], axis=1).to(device)
            y_pred, logits, _, c_pred = m_lem(sentence_batch, concept_labels)
            y_pred = y_pred.squeeze()
            concept_loss = 0
            for i in range(n_concepts):
                concept_loss += concept_form(c_pred[:,i,:], concept_labels.view(-1, n_concepts, concept_states[i])[:,i,:])
            concept_loss /= n_concepts            
            y = y.to(device)#.to(torch.long)
            task_loss = task_form(y_pred, y)  
            running_task_loss += task_loss.item()
            running_concept_loss += concept_loss.item()
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



# ------------------------------------------------------------------------------------- CBM --------------------------------------------------------------------------------------


@torch.no_grad()
def evaluate_cbm(loaded_set, encoder, concept_classifier, task_classifier, model, n_labels, n_concepts, concept_states, 
                 max_length, concept_names, concept_form=None, task_form=None, explain=False, device='cuda', mlp_head=False):
    model.eval()
    softmax = nn.Softmax(dim=-1)
    running_task_loss = 0
    running_concept_loss = 0
    task_preds = torch.zeros(1).to(device)
    concept_preds = torch.zeros(1,n_concepts).to(device)
    true_concepts = torch.zeros(1,n_concepts).to(device)
    pred_logits = torch.zeros(1, n_concepts*concept_states[0], n_labels).to(device)        
    pred_rules = []
    original_sentences = torch.zeros(1,max_length).to(device)
    real_labels = torch.zeros(1)
    concept_probs = torch.zeros(1, n_concepts, concept_states[0]).to(device)
    for sentence_batch in loaded_set:
        y = torch.Tensor(sentence_batch['labels'])
        emb = encoder(sentence_batch['input_ids'].squeeze().to(torch.long).to(device), 
                              sentence_batch['attention_mask'].squeeze().to(torch.long).to(device), 
                              output_hidden_states=True).hidden_states[-1][:,0,:]
        original_sentences = torch.cat([original_sentences, sentence_batch['input_ids'].to(device)])
        concept_labels = torch.cat([torch.nn.functional.one_hot(sentence_batch[f'{name}'].to(torch.long), concept_states[idx]).to(torch.float) for idx, name in enumerate(concept_names)], axis=1).to(device)

        true_concepts = torch.cat([true_concepts, concept_labels.view(-1, n_concepts, concept_states[0]).argmax(-1)])
        c_logit = concept_classifier(emb).view(-1, n_concepts, concept_states[0])
        c_pred = softmax(c_logit)
        y_pred = task_classifier(c_pred.flatten(start_dim=1)).squeeze()
        concept_probs = torch.cat([concept_probs, c_pred])
            
        if explain:
            weight_matrix = task_classifier[0].weight.data
            if mlp_head:
                logits = c_pred.flatten(start_dim=1).unsqueeze(1).repeat(1, n_labels, 1) # we compute some tensor just to be compliant with the other case, but we don't use it
            else:
                logits = c_pred.flatten(start_dim=1).unsqueeze(1).repeat(1, n_labels, 1) * weight_matrix.unsqueeze(0).repeat(c_pred.shape[0],1,1)
            pred_logits = torch.cat([pred_logits, logits.permute(0,2,1)])
            
        if concept_form!=None:
            concept_loss = 0
            for i in range(n_concepts):
                concept_loss += concept_form(c_logit[:,i,:], concept_labels.view(-1, n_concepts, concept_states[i])[:,i,:])
            concept_loss /= n_concepts
            running_concept_loss += concept_loss
            
        concept_preds = torch.cat([concept_preds, c_pred.argmax(-1)])
        y_pred = y_pred.squeeze()
        
        if task_form!=None:
            y = y.to(torch.long)
            running_task_loss += task_form(y_pred, y.to(device))  

        y_pred = y_pred.argmax(-1)
        task_preds = torch.cat([task_preds, y_pred])
        real_labels = torch.cat([real_labels, y])
    model.train()

    if explain:
        return task_preds[1:], real_labels[1:] , concept_preds[1:,:], \
               true_concepts[1:,:], pred_logits[1:,:,:], original_sentences[1:,:], concept_probs[1:,:,:]
    else:
        return running_task_loss.item()/len(loaded_set), running_concept_loss.item()/len(loaded_set), \
               task_preds[1:], real_labels[1:] , concept_preds[1:,:], true_concepts[1:,:]



def train_cbm(embedding_size, model_type, loaded_train, loaded_val, loaded_test, max_length, concept_names, labeler, explain,
              tokenizer, lr, epochs, n_labels, n_concepts, concept_states, step_size, gamma, lambda_coeff, device='cuda'):
    
    encoder = BertForSequenceClassification.from_pretrained(tokenizer).to(device)
    for param in encoder.bert.parameters():
        param.requires_grad = False
    for param in encoder.bert.encoder.layer[-1].parameters():
        param.requires_grad = True 
    concept_classifier = nn.Sequential(nn.Linear(embedding_size, n_concepts*concept_states[0]))
    softmax = nn.Softmax(dim=-1)

    if 'cbm_ll' in model_type:
        task_classifier = nn.Sequential(nn.Linear(n_concepts*concept_states[0], n_labels, bias=False))
    else:
        task_classifier = nn.Sequential(nn.Linear(n_concepts*concept_states[0], n_concepts*concept_states[0]), 
                                        nn.ReLU(), 
                                        nn.Linear(n_concepts*concept_states[0], n_labels, bias=False))
        
    model = nn.Sequential(encoder, concept_classifier, task_classifier).to(device)
    print('Number of trainable parameters:', sum(p.numel() if p.requires_grad==True else 0 for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    concept_form =  torch.nn.CrossEntropyLoss()
    task_form = torch.nn.CrossEntropyLoss()
    train_task_losses = []
    train_concept_losses = []
    val_task_losses = []
    val_concept_losses = []
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    model.train()
    for epoch in range(epochs):
        running_task_loss = 0
        running_concept_loss = 0

        for sentence_batch in tqdm(loaded_train):
            optimizer.zero_grad()
            y = torch.Tensor(sentence_batch['labels'])
            emb = encoder(sentence_batch['input_ids'].squeeze().to(torch.long).to(device), 
                                  sentence_batch['attention_mask'].squeeze().to(torch.long).to(device), 
                                  output_hidden_states=True).hidden_states[-1][:,0,:]
            concept_labels = torch.cat([torch.nn.functional.one_hot(sentence_batch[f'{name}'].to(torch.long), concept_states[idx]).to(torch.float) for idx, name in enumerate(concept_names)], axis=1).to(device)
            
            c_logit = concept_classifier(emb).view(-1, n_concepts, concept_states[0])
            c_pred = softmax(c_logit)
            y_pred = task_classifier(c_pred.flatten(start_dim=1)).squeeze()
            
            concept_loss = 0
            for i in range(n_concepts):
                concept_loss += concept_form(c_logit[:,i,:], concept_labels.view(-1, n_concepts, concept_states[i])[:,i,:])
            concept_loss /= n_concepts
            
            y = y.to(device).to(torch.long)
            task_loss = task_form(y_pred, y)  
            running_task_loss += task_loss.item()
            running_concept_loss += concept_loss.item()
            loss = concept_loss + lambda_coeff * task_loss 
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_task_losses.append(running_task_loss/len(loaded_train))
        train_concept_losses.append(running_concept_loss/len(loaded_train))   
        
        params = {
        'loaded_set':loaded_val,
        'encoder':encoder, 
        'concept_classifier':concept_classifier, 
        'task_classifier':task_classifier, 
        'model':model, 
        'n_labels':n_labels,
        'n_concepts':n_concepts, 
        'concept_states':concept_states, 
        'max_length':max_length, 
        'device':device,
        'concept_form':concept_form,
        'task_form':task_form,
        'concept_names': concept_names,
        'explain': explain,
        'mlp_head': 'mlp' in model_type
        }
        
        val_task_loss, val_concept_loss, _, _, _, _ = evaluate_cbm(**params)
        val_task_losses.append(val_task_loss)
        val_concept_losses.append(val_concept_loss)

    params['loaded_set'] = loaded_test
    params['explain'] = True
    y_preds, y, c_preds, c_true, _, _, _ = evaluate_cbm(**params)

    return train_task_losses, train_concept_losses, val_task_losses, val_concept_losses, y_preds, y, c_preds, c_true



# -------------------------------------------------------------------------------------------- E2E -----------------------------------------------------------------------------------------------



@torch.no_grad()
def evaluate_e2e(loaded_set, encoder, task_classifier, model, n_labels, max_length, task_form=None, device='cuda'):
    model.eval()
    softmax = nn.Softmax(dim=-1)
    running_task_loss = 0
    task_preds = torch.zeros(1).to(device)
    original_sentences = torch.zeros(1,max_length).to(device)
    real_labels = torch.zeros(1)
    for sentence_batch in loaded_set:
        y = torch.Tensor(sentence_batch['labels'])
        emb = encoder(sentence_batch['input_ids'].squeeze().to(torch.long).to(device), 
                              sentence_batch['attention_mask'].squeeze().to(torch.long).to(device), 
                              output_hidden_states=True).hidden_states[-1][:,0,:]
        original_sentences = torch.cat([original_sentences, sentence_batch['input_ids'].to(device)])
    
        y_pred = task_classifier(emb).squeeze()

        y_pred = y_pred.squeeze()
        
        if task_form!=None:
            y = y.to(torch.long)
            running_task_loss += task_form(y_pred, y.to(device))  

        y_pred = y_pred.argmax(-1)
        task_preds = torch.cat([task_preds, y_pred])
        real_labels = torch.cat([real_labels, y])
    model.train()

    return running_task_loss.item()/len(loaded_set), task_preds[1:], real_labels[1:]



def train_e2e(embedding_size, loaded_train, loaded_val, loaded_test, max_length,
              tokenizer, lr, epochs, n_labels, step_size, gamma, device='cuda'):
    
    encoder = BertForSequenceClassification.from_pretrained(tokenizer).to(device)
    for param in encoder.bert.parameters():
        param.requires_grad = False
    for param in encoder.bert.encoder.layer[-1].parameters():
        param.requires_grad = True 
    
    task_classifier = nn.Sequential(nn.Linear(embedding_size, n_labels), nn.Softmax(dim=-1))
        
    model = nn.Sequential(encoder, task_classifier).to(device)
    print('Number of trainable parameters:', sum(p.numel() if p.requires_grad==True else 0 for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    task_form = torch.nn.CrossEntropyLoss()
    train_task_losses = []
    val_task_losses = []
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    model.train()
    for epoch in range(epochs):
        running_task_loss = 0

        for sentence_batch in tqdm(loaded_train):
            optimizer.zero_grad()
            y = torch.Tensor(sentence_batch['labels'])
            emb = encoder(sentence_batch['input_ids'].squeeze().to(torch.long).to(device), 
                                  sentence_batch['attention_mask'].squeeze().to(torch.long).to(device), 
                                  output_hidden_states=True).hidden_states[-1][:,0,:]

            y_pred = task_classifier(emb).squeeze()

            y = y.to(device).to(torch.long)
            task_loss = task_form(y_pred, y)  
            running_task_loss += task_loss.item()
            loss = task_loss 
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_task_losses.append(running_task_loss/len(loaded_train))
        
        params = {
        'loaded_set':loaded_val,
        'encoder':encoder, 
        'task_classifier':task_classifier, 
        'model':model, 
        'n_labels':n_labels,
        'max_length':max_length, 
        'device':device,
        'task_form':task_form,
        }        
        val_task_loss, _, _ = evaluate_e2e(**params)
        val_task_losses.append(val_task_loss)

    params['loaded_set'] = loaded_test
    _, y_preds, y = evaluate_e2e(**params)
       
    return encoder, task_classifier, model, train_task_losses, val_task_losses, y_preds, y


