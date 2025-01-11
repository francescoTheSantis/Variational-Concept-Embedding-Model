import argparse
from toy_loaders import *
from models import *
from training import *
from utilities import *
import os
import csv
import pandas as pd
from text_loaders import *
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
    elif args.dataset == 'cebab':
        train_dataset = CEBABDataset('train')
        val_dataset = CEBABDataset('validation')
        test_dataset = CEBABDataset('test')
        loaded_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        loaded_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        loaded_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        loaders_path = f'{args.output_dir}'.replace('results','data')
        pass
    elif args.dataset == 'imdb':
        train_dataset = IMDBDataset('train')
        val_dataset = IMDBDataset('validation')
        test_dataset = IMDBDataset('test')
        loaded_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        loaded_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        loaded_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        loaders_path = f'{args.output_dir}'.replace('results','data')
        pass
    elif args.dataset == 'mnist_add':
        loaders_path = f'{args.output_dir}'.replace('results','data')
        loaded_train, loaded_val, loaded_test = MNIST_addition_loader(args.batch_size, loaders_path, val_size=0.1, seed=42) # the seed is fixed for the dataset creation
        # process the loaded data using ResNet18 such that we do not require to pass the images in the ResNet18 model multiple times.
        # this is done to speed up the training process.
        pass

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
    elif args.dataset == 'cebab':
        in_features = 384
        n_concepts = 4
        n_labels = 2
    elif args.dataset == 'imdb':
        in_features = 384
        n_concepts = 8
        n_labels = 2
    elif args.dataset == 'mnist_add':
        in_features = 768
        n_concepts = 10
        n_labels = 19

    if args.model == 'e2e':
        classifier = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_labels)
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
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_concepts),
            nn.Sigmoid()
        ) 
    elif args.model == 'cbm_mlp':
        classifier = nn.Sequential(
            nn.Linear(n_concepts, 16),
            nn.ReLU(),
            nn.Linear(16, n_labels)
        )
        concept_encoder = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_concepts),
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
        'step_size': 100,
        'gamma': 0.1,
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
        epss = [0, 0.1, 0.25, 0.5, 0.75, 1]
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
            epss = [0, 0.25, 0.5, 0.75, 1]
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument('--dataset', type=str, help='The name of the dataset')
    parser.add_argument('--emb_size', type=int, default=16, help='The size of the concept embeddings')  
    parser.add_argument('--model', type=str, default='linear', help='The model to use for the experiment')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size to use for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to run')
    parser.add_argument('--device', type=str, default='cuda', help='The device to use for training')
    parser.add_argument('--output_dir', type=str, required=True, help='The output directory to save the results')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--sampling', type=bool, default=False, help='Whether to sample from the distribution or take the MAP (only for AA_CEM)')
    args = parser.parse_args()
    
    main(args)