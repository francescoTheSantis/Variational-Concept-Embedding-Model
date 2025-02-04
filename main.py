from src.models.blackbox import BlackboxModel
from src.models.cbm import ConceptBottleneckModel
from src.models.v_cem import VariationalConceptEmbeddingModel

from src.trainer import Trainer
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import wandb
from src.utilities import set_seed
from pytorch_lightning.loggers import WandbLogger, CSVLogger

@hydra.main(config_path="config", config_name="sweep")
def main(cfg: DictConfig) -> None:

    wandb.init(project=cfg.wandb.project,
               entity=cfg.wandb.entity, 
               name=f"{cfg.model.metadata.name}_{cfg.dataset.metadata.name}_{cfg.seed}")
    wandb_logger = WandbLogger(project=cfg.wandb.project, 
                               entity=cfg.wandb.entity, 
                               name=f"{cfg.model.metadata.name}_{cfg.dataset.metadata.name}_{cfg.seed}")
    csv_logger = CSVLogger("logs/", name="experiment_metrics")

    print("Configuration Parameters:")
    for key, value in cfg.items():
        print(f"{key}: {value}")
    print('\n')

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
    if cfg.model.metadata.name != 'blackbox':
        # Perform interventions experiment
        intervention_df = trainer.interventions(loaded_test)

        # Save the intervention results
        log_dir = csv_logger.log_dir
        intervention_df.to_csv(f"{log_dir}/interventions.csv", index=False)

if __name__ == "__main__":
    main()