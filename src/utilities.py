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
    # Ensure predictions and labels are 2D tensors
    assert predictions.dim() == 2 and labels.dim() == 2, "Both tensors must be 2D"
    
    # Find mismatched indices if all_entries is False, select all otherwise
    if all_entries:
        mismatched_indices = (torch.ones_like(hard_predictions)).nonzero(as_tuple=False)
    else:
        mismatched_indices = (hard_predictions != labels).nonzero(as_tuple=False)

    # Randomly select mismatched indices based on the given probability
    num_mismatches = mismatched_indices.size(0)

    if repeat!=None:
        mask_list = []
        for i in range(repeat):
            mask = torch.rand(num_mismatches) < probability
            idxs_mask = mismatched_indices[mask]
            mask = torch.zeros_like(predictions)
            for index in idxs_mask:
                mask[index[0], index[1]] = 1
            mask_list.append(mask)
        mask_tensor = torch.stack(mask_list, dim=-1)
        intervened = labels * mask + predictions * (1 - mask)
        mask = mask_tensor
    else:
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