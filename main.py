#from src.loaders.toy_loaders import *
#from src.loaders.text_loaders import *
#from src.loaders.image_loaders import *
#from src.models.v_cem import *
#from src.models.cbm import *
#from src.models.cem import *

from src.models.blackbox import BlackboxModel
from src.models.cbm import ConceptBottleneckModel

from src.utilities import *
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

    ######  Load the data ######
    loaded_train, loaded_val, loaded_test = instantiate(cfg.dataset.loader)

    ###### Instantiate the model ######
    model = instantiate(cfg.model.params)

    ####### Training ########
    # Initialize the trainer
    trainer = Trainer(model, cfg, wandb_logger, csv_logger)
    trainer.build_trainer()
    
    # Train the model
    trainer.train(loaded_train, loaded_val)

    ###### Test ########
    # Test the model on the test-set
    trainer.test(loaded_test)

    ##### Intervetions ######
    '''
    task_f1, task_acc = f1_acc_metrics(y, y_preds)
    concept_f1, concept_acc = 0, 0
    if cfg.model != 'e2e':
        for i in range(n_concepts):
            f1, acc = f1_acc_metrics(c_true[i], torch.where(c_preds[i]>0.5,1,0))
            concept_f1 += f1
            concept_acc += acc
        concept_f1 /= n_concepts
        concept_acc /= n_concepts
    print(f'Task F1: {task_f1}, Task Accuracy: {task_acc}')
    print(f'Concept F1: {concept_f1}, Concept Accuracy: {concept_acc}')
    csv_file_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['type', 'f1', 'accuracy'])
        writer.writerow(['task', task_f1, task_acc])
        writer.writerow(['concept', concept_f1, concept_acc])
    print(f'Metrics saved to {csv_file_path}')

    # now we gradually increase the noise in the input features and perform interventions
    if cfg.model != 'e2e':
        intervention_df = pd.DataFrame(columns=['noise', 'p_int', 'f1', 'accuracy'])
        params = {
            'model': args.model,
            'concept_encoder': concept_encoder,
            'classifier': classifier,
            'loaded_set': loaded_test,
            'n_concepts': n_concepts,
            'emb_size': args.emb_size,
            'concept_form': nn.BCELoss(),
            'task_form': nn.CrossEntropyLoss(),
            'device': 'cuda',
            'n_labels': n_labels
        }
        interventions_dir = os.path.join(output_dir, 'interventions')
        if not os.path.exists(interventions_dir):
            os.makedirs(interventions_dir)
        csv_file_path = os.path.join(interventions_dir, 'metrics.csv')
        epss = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
        p_ints = np.arange(0, 1.1, 0.1)
        for eps in epss:
            params['corruption'] = eps
            for p_int in p_ints:
                params['intervention_prob'] = p_int
                _, _, _, y_preds, y, c_preds, c_true, _ = evaluate(**params)
                task_f1, task_acc = f1_acc_metrics(y, y_preds)
                # create a dictionary with the results
                intervention_results = {'noise': eps, 'p_int': p_int, 'f1': task_f1, 'accuracy': task_acc}
                intervention_df = intervention_df.append(intervention_results, ignore_index=True)
        intervention_df.to_csv(csv_file_path, index=False)
    '''

    '''
    if args.model=='aa_cem':
        concept_encoder.embedding_interventions = False
        params['concept_encoder'] = concept_encoder
        csv_file_path = os.path.join(interventions_dir, 'metrics_concept_score_intervention.csv')
        for eps in epss:
            params['corruption'] = eps
            for p_int in p_ints:
                params['intervention_prob'] = p_int
                _, _, _, y_preds, y, c_preds, c_true, _ = evaluate(**params)
                task_f1, task_acc = f1_acc_metrics(y, y_preds)
                # create a dictionary with the results
                intervention_results = {'noise': eps, 'p_int': p_int, 'f1': task_f1, 'accuracy': task_acc}
                intervention_df = intervention_df.append(intervention_results, ignore_index=True)
        intervention_df.to_csv(csv_file_path, index=False)            
    '''

if __name__ == "__main__":
    main()