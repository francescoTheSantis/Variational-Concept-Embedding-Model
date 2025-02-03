import argparse
from toy_loaders import *
from text_loaders import *
from models.v_cem import *
from training import *
from utilities import *
import os
import csv
import pandas as pd
#from text_loaders import *
from image_loaders import *

def main(args):
    set_seed(args.seed)
    output_dir = os.path.join(args.output_dir, str(args.seed), args.dataset, args.model) 
    if not os.path.exists(output_dir):
        print('Results folder created')
        os.makedirs(output_dir)
    '''
    else:
        if not os.listdir(output_dir):
            print('Results folder exists but is empty.')
        else:
            print('Results folder exists and is not empty. The experiment has already been executed!')
            return
    '''
    print('Results will be saved to:', output_dir)

    if args.dataset in ['xor', 'and', 'or', 'trigonometry', 'dot']:
        loaded_train, loaded_val, loaded_test = Toy_DataLoader(args.dataset, args.batch_size, 800, 100, 100).get_data_loaders()
    elif args.dataset == 'mnist_add':
        #loaders_path = f'{args.output_dir}'.replace('results','data')
        loaded_train, loaded_val, loaded_test = MNIST_addition_loader(args.batch_size, val_size=0.1, seed=42) # the seed is fixed for the dataset creation
        # process the loaded data using ResNet18 such that we do not require to pass the images in the ResNet18 model multiple times.
        # this is done to speed up the training process.
        E_extr = EmbeddingExtractor(loaded_train, loaded_val, loaded_test, device=args.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
    elif args.dataset == 'cub':
        loaded_train, loaded_val, loaded_test = CUB200_loader(args.batch_size, val_size=0.1, seed=42, dataset=f'{args.root}/data/cub', num_workers=3, pin_memory=True, augment=True, shuffle=True)
        E_extr = EmbeddingExtractor(loaded_train, loaded_val, loaded_test, device=args.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
    elif args.dataset == 'celeba':
        concept_names = ['Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick', 'High_Cheekbones', 'Heavy_Makeup', 'Wavy_Hair', 'Oval_Face', 'Pointy_Nose', 'Arched_Eyebrows', 'Big_Lips'] # 'Wearing_Lipstick', 'Heavy_Makeup'
        class_attributes = ['Attractive']        
        loaded_train, loaded_val, loaded_test = CelebA_loader(args.batch_size, val_size=0.1, seed = 42, dataset=f'{args.root}/data', class_attributes=class_attributes, concept_names=concept_names, num_workers=3, pin_memory=True, shuffle=True)
        #loaded_train, loaded_val, loaded_test = CUB200_loader(args.batch_size, val_size=0.1, seed=42, dataset=f'{args.root}/data/cub', num_workers=3, pin_memory=True, augment=True, shuffle=True)
        E_extr = EmbeddingExtractor(loaded_train, loaded_val, loaded_test, device=args.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
    elif args.dataset == 'cebab':
        loaded_train = CEBABDataset(args.root, 'train', model_name='all-MiniLM-L6-v2')
        loaded_val = CEBABDataset(args.root, 'validation', model_name='all-MiniLM-L6-v2')
        loaded_test = CEBABDataset(args.root, 'test', model_name='all-MiniLM-L6-v2')
        E_extr = EmbeddingExtractor_text(loaded_train, loaded_val, loaded_test, args.batch_size, device=args.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
    elif args.dataset == 'imdb':
        loaded_train = IMDBDataset(args.root, 'train', model_name='all-MiniLM-L6-v2')
        loaded_val = IMDBDataset(args.root, 'validation', model_name='all-MiniLM-L6-v2')
        loaded_test = IMDBDataset(args.root, 'test', model_name='all-MiniLM-L6-v2')
        E_extr = EmbeddingExtractor_text(loaded_train, loaded_val, loaded_test, args.batch_size, device=args.device)
        loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()


    if args.dataset in ['xor', 'and', 'or']:
        in_features = 2
        n_concepts = 2
        n_labels = 2
    elif args.dataset == 'trigonometry':
        in_features = 7
        n_concepts = 3  
        n_labels = 2
    elif args.dataset == 'dot':
        in_features = 4
        n_concepts = 2
        n_labels = 2  
    elif args.dataset == 'mnist_add':
        in_features = 512
        n_concepts = 10
        n_labels = 20
    elif args.dataset == 'cub':
        in_features = 512
        n_concepts = 112
        n_labels = 200
    elif args.dataset == 'celeba':
        in_features = 512
        n_concepts = len(concept_names)
        n_labels = 2
    elif args.dataset == 'cebab':
        in_features = 384
        n_concepts = 4
        n_labels = 2
    elif args.dataset == 'imdb':
        in_features = 384
        n_concepts = 8
        n_labels = 2

    if args.model == 'e2e':
        classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, n_labels)
        )
        concept_encoder = None
    elif args.model == 'cem':
        classifier = nn.Sequential(
            nn.Linear(args.emb_size*n_concepts, args.emb_size*n_concepts),
            nn.ReLU(),
            nn.Linear(args.emb_size*n_concepts, n_labels)
        )        
        concept_encoder = ConceptEmbedding(in_features, n_concepts, args.emb_size)
    elif args.model == 'aa_cem':
        classifier = nn.Sequential(
            nn.Linear(args.emb_size*n_concepts, args.emb_size*n_concepts),
            nn.ReLU(),
            nn.Linear(args.emb_size*n_concepts, n_labels)
        )                        
        concept_encoder = AA_CEM(in_features, n_concepts, args.emb_size, True, args.sampling)
    elif args.model == 'cbm_linear':
        classifier = nn.Sequential(
            nn.Linear(n_concepts, n_labels)
        )
        concept_encoder = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, n_concepts),
            nn.Sigmoid()
        ) 
    elif args.model == 'cbm_mlp':
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

    params = {
        'model': args.model,
        'loaded_train': loaded_train,
        'loaded_val': loaded_val,
        'loaded_test': loaded_test,
        'concept_encoder': concept_encoder,
        'classifier': classifier,
        'lr': args.lr,
        'epochs': args.epochs,
        'n_concepts': n_concepts,
        'step_size': 20,
        'gamma': 0.5,
        'device': args.device,
        'emb_size': args.emb_size,
        'test': False,
        'n_labels': n_labels,
        'patience': args.patience,
        'sampling': args.sampling,
        'folder': output_dir
    }
    
    concept_encoder, classifier, train_task_losses, train_concept_losses, D_kl_losses, val_task_losses, val_concept_losses, D_kl_losses_val, y_preds, y, c_preds, c_true, c_emb = train(**params)

    plot_training_curves(train_task_losses, val_task_losses, train_concept_losses, val_concept_losses, D_kl_losses, D_kl_losses_val, output_dir)
    task_f1, task_acc = f1_acc_metrics(y, y_preds)
    concept_f1, concept_acc = 0, 0
    if args.model != 'e2e':
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
    if args.model != 'e2e':
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument('--root', type=str, default=None, help='The root directory for the dataset')
    parser.add_argument('--dataset', type=str, help='The name of the dataset')
    parser.add_argument('--emb_size', type=int, default=16, help='The size of the concept embeddings')  
    parser.add_argument('--model', type=str, default='linear', help='The model to use for the experiment')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size to use for training')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to run')
    parser.add_argument('--device', type=str, default='cuda', help='The device to use for training')
    parser.add_argument('--output_dir', type=str, required=True, help='The output directory to save the results')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--sampling', type=bool, default=False, help='Whether to sample from the distribution or take the MAP (only for AA_CEM)')
    args = parser.parse_args()
    
    main(args)