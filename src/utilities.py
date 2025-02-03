import os
from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np
#import scienceplots
import warnings
warnings.filterwarnings("ignore")
import random

#plt.style.use(['science', 'ieee'])

def set_seed(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        
def D_kl_gaussian(mu_q, logvar_q, mu_p, with_variance=True):
    if with_variance:
        value = -0.5 * torch.sum(1 + logvar_q - (mu_q - mu_p).pow(2) - logvar_q.exp(), dim=-1)
    else:
        value = torch.sum((mu_q - mu_p).pow(2), dim=-1)
    return value.mean()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, separate=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        if separate:
            self.min_task = float('inf')
            self.min_concept = float('inf')
            self.min_kl = float('inf')
        else:
            self.min_loss = float('inf')
        self.best_iteration = True
        self.separate = separate

    def early_stop_separate(self, task, concept, kl):
        if (task-self.min_task)<-self.min_delta or (concept-self.min_concept)<-self.min_delta or (kl-self.min_kl)<-self.min_delta:
            self.min_task = task if task < self.min_task else self.min_task
            self.min_concept = concept if concept < self.min_concept else self.min_concept
            self.min_kl = kl if kl < self.min_kl else self.min_kl
            self.counter = 0
            self.best_iteration = True
        else:
            self.counter += 1
            self.best_iteration = False
            if self.counter >= self.patience:
                return True
        return False

    def early_stop_sum(self, task, concept, kl):
        loss = task + concept + kl
        if (loss-self.min_loss)<-self.min_delta:
            self.min_loss = loss
            self.counter = 0
            self.best_iteration = True
        else:
            self.counter += 1
            self.best_iteration = False
            if self.counter >= self.patience:
                return True
        return False
    
    def early_stop(self, task, concept, kl):
        if self.separate:
            return self.early_stop_separate(task, concept, kl)
        else:
            return self.early_stop_sum(task, concept, kl)
        

def get_intervened_concepts_predictions(predictions, labels, probability, return_index=False):
    
    hard_predictions = torch.where(predictions > 0.5, 1, 0)
    # Ensure predictions and labels are 2D tensors
    assert predictions.dim() == 2 and labels.dim() == 2, "Both tensors must be 2D"
    
    # Find mismatched indices
    mismatched_indices = (hard_predictions != labels).nonzero(as_tuple=False)

    # Randomly select mismatched indices based on the given probability
    num_mismatches = mismatched_indices.size(0)
    mask = torch.rand(num_mismatches) < probability
    idxs_mask = mismatched_indices[mask]
    mask = torch.zeros_like(predictions)
    for index in idxs_mask:
        mask[index[0], index[1]] = 1
    intervened = labels * mask + predictions * (1 - mask)

    if return_index:
        return mask, intervened
    else:
        return intervened

def plot_training_curves(train_task_losses, val_task_losses, train_concept_losses, val_concept_losses, d_kl, val_d_kl, output_folder=None):

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))

    # Plot task training and validation losses
    axs[0].plot(train_task_losses, label='Train Task Loss')
    axs[0].plot(val_task_losses, label='Validation Task Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Task Training and Validation Loss')
    axs[0].legend()

    # Plot concept training and validation losses
    axs[1].plot(train_concept_losses, label='Train Concept Loss')
    axs[1].plot(val_concept_losses, label='Validation Concept Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Concept Training and Validation Loss')
    axs[1].legend()

    # Plot KL divergence training and validation losses
    axs[2].plot(d_kl, label='Train KL Divergence')
    axs[2].plot(val_d_kl, label='Validation KL Divergence')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].set_title('KL Divergence Training and Validation Loss')
    axs[2].legend()

    # Save the figure
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_validation_losses.pdf'))
    plt.show()  # Show the plot
    plt.close()

def f1_acc_metrics(y_true, y_pred):
    # Convert PyTorch tensors to lists if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy().tolist()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy().tolist()
    
    # Calculate the F1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    # Calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return f1, accuracy

