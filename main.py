from src.trainer import Trainer
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, get_class
from src.utilities import set_seed, set_loggers
from src.metrics import concept_alignment_score
import torch
import os
import pandas as pd

@hydra.main(config_path="config", config_name="sweep")
def main(cfg: DictConfig) -> None:

    # Initialize the wandb logger
    wandb_logger, csv_logger = set_loggers(cfg)

    print("Configuration Parameters:")
    for key, value in cfg.items():
        print(f"{key}: {value}")
    print('\n')

    # Set the seed
    set_seed(cfg.seed)

    ###### Load the data ######
    data_path = os.path.join(cfg.root, 'stored_tensors', cfg.dataset.metadata.name)
    train_path = f"{data_path}/train.pt"
    val_path = f"{data_path}/val.pt"
    test_path = f"{data_path}/test.pt"
    # If the data have been preprocessed, load the preprocessed data
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print('Loading pre-processed data...')
        loaded_train = torch.load(f"{data_path}/train.pt")
        loaded_val = torch.load(f"{data_path}/val.pt")
        loaded_test = torch.load(f"{data_path}/test.pt")
    # Otherwise, preprocess the data and then store them
    else:
        print('Preprocessing data...')
        loaded_train, loaded_val, loaded_test = instantiate(cfg.dataset.loader)
        os.makedirs(data_path, exist_ok=True)
        torch.save(loaded_train, train_path)
        torch.save(loaded_val, val_path)
        torch.save(loaded_test, test_path)

    ###### Instantiate the model ######
    model = instantiate(cfg.model.params)

    ###### Training ######
    # Initialize the trainer
    trainer = Trainer(model, cfg, wandb_logger, csv_logger)
    trainer.build_trainer()

    # Train the model
    trainer.train(loaded_train, loaded_val)

    # Load the best model
    model_class = get_class(cfg.model.params._target_)
    model_kwargs = {k: v for k, v in cfg.model.params.items() if k not in ["_target_"]}
    model = model_class.load_from_checkpoint(trainer.trainer.checkpoint_callback.best_model_path, **model_kwargs)
    trainer.model = model

    ###### Test ######
    # Test the model on the test-set
    trainer.test(loaded_test)

    ###### Intervetions ######
    if model.has_concepts:
        # Perform interventions for different noise levels
        intervention_df = trainer.interventions(loaded_test)
        log_dir = csv_logger.log_dir
        intervention_df.to_csv(f"{log_dir}/interventions.csv", index=False)

        # Store the latent representations
        latents, concept_prediction, concept_ground_truth, task_ground_truth = trainer.get_latents(loaded_test)
        torch.save(latents, f"{log_dir}/latents.pt")
        torch.save(concept_ground_truth, f"{log_dir}/concept_ground_truth.pt")
        torch.save(concept_prediction, f"{log_dir}/concept_prediction.pt")
        torch.save(task_ground_truth, f"{log_dir}/task_ground_truth.pt")

    # Close the wandb logger if it is used
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()