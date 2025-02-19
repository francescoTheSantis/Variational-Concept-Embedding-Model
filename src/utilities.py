import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
from omegaconf import DictConfig, OmegaConf
from time import time
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb

def set_seed(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_intervened_concepts_predictions(predictions, labels, probability, return_index=False, all_entries=False, repeat=None):
    
    hard_predictions = torch.where(predictions > 0.5, 1, 0)

    if repeat!=None:
        hard_predictions = hard_predictions.unsqueeze(-1).expand(-1, -1, repeat)  
        labels = labels.unsqueeze(-1).expand(-1, -1, repeat)   
        predictions = predictions.unsqueeze(-1).expand(-1, -1, repeat)   
        
    # Find mismatched indices if all_entries is False, select all otherwise
    if all_entries:
        mismatched_mask = (torch.ones_like(hard_predictions))#.nonzero(as_tuple=False)
    else:
        mismatched_mask = (hard_predictions != labels)#.nonzero(as_tuple=False)

    '''
    # Randomly select mismatched indices based on the given probability
    num_mismatches = mismatched_indices.size(0)

    mask = torch.rand(num_mismatches) < probability
    idxs_mask = mismatched_indices[mask]
    mask = torch.zeros_like(predictions)
    for index in idxs_mask:
        if repeat!=None:
            mask[index[0], index[1], index[2]] = 1
        else:
            mask[index[0], index[1]] = 1
    intervened = labels * mask + predictions * (1 - mask)
    '''

    # Generate a probability mask of the same shape
    random_mask = torch.rand_like(predictions, dtype=torch.float)

    # Apply probability threshold only on mismatched elements
    mask = (random_mask < probability) & mismatched_mask
    mask = mask.int()

    # Apply intervention
    intervened = labels * mask + predictions * (1-mask)

    if return_index:
        return mask, intervened
    else:
        return intervened

def set_loggers(cfg):
    name = f"seed{cfg.seed}.{int(time())}"
    
    group_format = (
        "{dataset}"
        "{model}"
        "{emb_size}"
        "{kl_penalty}"
    )

    group = group_format.format(**parse_hyperparams(cfg))

    wandb_logger = WandbLogger(project=cfg.wandb.project, 
                               entity=cfg.wandb.entity, 
                               name=name,
                               group=group)
    
    csv_logger = CSVLogger("logs/", 
                           name="experiment_metrics")

    return wandb_logger, csv_logger

        
def parse_hyperparams(cfg: DictConfig):
    hyperparams = {
        "dataset": cfg.dataset.metadata.name,
        "model": cfg.model.metadata.name,
        "seed": cfg.seed,
        "emb_size": cfg.c_emb_size,
        "kl_penalty": cfg.kl_penalty,
        "hydra_cfg": OmegaConf.to_container(cfg),
    }
    return hyperparams