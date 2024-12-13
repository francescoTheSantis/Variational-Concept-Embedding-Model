import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np
#import scienceplots
import warnings
warnings.filterwarnings("ignore")

#plt.style.use(['science', 'ieee'])

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

def D_kl_gaussian(mu_q, logvar_q, mu_p):
    value = -0.5 * torch.sum(1 + logvar_q - (mu_q - mu_p).pow(2) - logvar_q.exp(), dim=-1)
    return value.mean()

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

