from src.trainer import Trainer
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.utilities import set_seed, set_loggers
import torch

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
    loaded_train, loaded_val, loaded_test = instantiate(cfg.dataset.loader)

    ###### Instantiate the model ######
    model = instantiate(cfg.model.params)

    ###### Training ######
    # Initialize the trainer
    trainer = Trainer(model, cfg, wandb_logger, csv_logger)
    trainer.build_trainer()

    # Train the model
    trainer.train(loaded_train, loaded_val)

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
        latents, concept_ground_truth = trainer.get_latents(loaded_test)
        torch.save(latents, f"{log_dir}/latents.pt")
        torch.save(concept_ground_truth, f"{log_dir}/concept_ground_truth.pt")

    # Close the wandb logger
    wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()