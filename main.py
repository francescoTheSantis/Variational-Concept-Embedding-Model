#from src.loaders.toy_loaders import *
#from src.loaders.text_loaders import *
#from src.loaders.image_loaders import *
#from src.models.v_cem import *
#from src.models.cbm import *
#from src.models.cem import *

from src.utilities import *
import os
import csv
import pandas as pd
import hydra
#from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
from hydra.utils import instantiate


@hydra.main(config_path="config", config_name="sweep")
def main(cfg: DictConfig) -> None:

    #wandb_logger = WandbLogger(project=cfg.wandb.project, entity=cfg.wandb.entity)

    set_seed(cfg.seed)
    output_dir = os.path.join(cfg.root, 'results', cfg.model, cfg.dataset, str(cfg.seed)) 
    if not os.path.exists(output_dir):
        print('Results folder created')
        os.makedirs(output_dir)
    print('Results will be saved to:', output_dir)

    # dataset is a dicitonary which contains the different data splits (train, val and test)
    # and the other parameters related to the dataset
    dataset = instantiate(cfg.dataset)

    '''
    if cfg.dataset in ['xor', 'and', 'or', 'trigonometry', 'dot']:
        loaded_train, loaded_val, loaded_test = Toy_DataLoader(cfg.dataset, cfg.batch_size, 800, 100, 100).get_data_loaders()
    elif cfg.dataset == 'mnist_add':
        loaded_train, loaded_val, loaded_test = MNIST_addition_loader(cfg.batch_size, val_size=0.1, seed=42)
        E_extr = EmbeddingExtractor(loaded_train, loaded_val, loaded_test, device=cfg.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
    elif cfg.dataset == 'cub':
        loaded_train, loaded_val, loaded_test = CUB200_loader(cfg.batch_size, val_size=0.1, seed=42, dataset=f'{cfg.root}/data/cub', num_workers=3, pin_memory=True, augment=True, shuffle=True)
        E_extr = EmbeddingExtractor(loaded_train, loaded_val, loaded_test, device=cfg.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
    elif cfg.dataset == 'celeba':
        concept_names = ['Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick', 'High_Cheekbones', 'Heavy_Makeup', 'Wavy_Hair', 'Oval_Face', 'Pointy_Nose', 'Arched_Eyebrows', 'Big_Lips'] # 'Wearing_Lipstick', 'Heavy_Makeup'
        class_attributes = ['Attractive']        
        loaded_train, loaded_val, loaded_test = CelebA_loader(cfg.batch_size, val_size=0.1, seed = 42, dataset=f'{cfg.root}/data', class_attributes=class_attributes, concept_names=concept_names, num_workers=3, pin_memory=True, shuffle=True)
        E_extr = EmbeddingExtractor(loaded_train, loaded_val, loaded_test, device=cfg.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
    elif cfg.dataset == 'cebab':
        loaded_train = CEBABDataset(cfg.root, 'train', model_name='all-MiniLM-L6-v2')
        loaded_val = CEBABDataset(cfg.root, 'validation', model_name='all-MiniLM-L6-v2')
        loaded_test = CEBABDataset(cfg.root, 'test', model_name='all-MiniLM-L6-v2')
        E_extr = EmbeddingExtractor_text(loaded_train, loaded_val, loaded_test, cfg.batch_size, device=cfg.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
    elif cfg.dataset == 'imdb':
        loaded_train = IMDBDataset(cfg.root, 'train', model_name='all-MiniLM-L6-v2')
        loaded_val = IMDBDataset(cfg.root, 'validation', model_name='all-MiniLM-L6-v2')
        loaded_test = IMDBDataset(cfg.root, 'test', model_name='all-MiniLM-L6-v2')
        E_extr = EmbeddingExtractor_text(loaded_train, loaded_val, loaded_test, cfg.batch_size, device=cfg.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()

    if cfg.dataset in ['xor', 'and', 'or']:
        in_features = 2
        n_concepts = 2
        n_labels = 2
    elif cfg.dataset == 'trigonometry':
        in_features = 7
        n_concepts = 3  
        n_labels = 2
    elif cfg.dataset == 'dot':
        in_features = 4
        n_concepts = 2
        n_labels = 2  
    elif cfg.dataset == 'mnist_add':
        in_features = 512
        n_concepts = 10
        n_labels = 20
    elif cfg.dataset == 'cub':
        in_features = 512
        n_concepts = 112
        n_labels = 200
    elif cfg.dataset == 'celeba':
        in_features = 512
        n_concepts = len(concept_names)
        n_labels = 2
    elif cfg.dataset == 'cebab':
        in_features = 384
        n_concepts = 4
        n_labels = 2
    elif cfg.dataset == 'imdb':
        in_features = 384
        n_concepts = 8
        n_labels = 2
    '''

    if cfg.model == 'e2e':
        classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, n_labels)
        )
        concept_encoder = None
    elif cfg.model == 'cem':
        model = ConceptEmbeddingModel(
            n_concepts=n_concepts,
            n_tasks=n_labels,
            emb_size=cfg.emb_size,
            training_intervention_prob=cfg.training_intervention_prob,
            task_loss_weight=cfg.task_loss_weight,
        )
    elif cfg.model == 'aa_cem':
        model = V_CEM(in_features, 
                      n_concepts, 
                      n_labels, 
                      cfg.emb_size, 
                      cfg.task_penalty, 
                      cfg.kl_penalty, 
                      cfg.p_int_train)

    elif cfg.model == 'cbm_linear':
        classifier = nn.Sequential(
            nn.Linear(n_concepts, n_labels)
        )
        concept_encoder = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, n_concepts),
            nn.Sigmoid()
        ) 
    elif cfg.model == 'cbm_mlp':
        classifier = nn.Sequential(
            nn.Linear(n_concepts, n_concepts),
            nn.ReLU(),
            nn.Linear(n_concepts, n_labels)
        )
        concept_encoder = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, n_concepts),
            nn.Sigmoid()
        ) 



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