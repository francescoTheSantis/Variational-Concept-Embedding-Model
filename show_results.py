import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import warnings
import torch
import os
import yaml
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

# Add path to the experiments
path = ''
exps = os.listdir(path)
exps_path = [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

performance = pd.DataFrame()

for exp in exps_path:
    d = {}
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/metrics.csv')  
    #cas_file = os.path.join(exp, 'logs/experiment_metrics/version_0/cas.csv')   
    print(exp)
    if os.path.exists(conf_file) and os.path.exists(result_file):
        try:
            with open(conf_file, 'r') as file:
                conf = yaml.safe_load(file)
            d['seed'] = conf['seed']
            d['dataset'] = conf['dataset']['metadata']['name']
            d['model'] = conf['model']['metadata']['name']

            with open(result_file, 'r') as file:
                result = pd.read_csv(file, header=0)

            d['task'] = result['test_task_acc'].iloc[-1]
            d['concept'] = result['test_concept_acc'].iloc[-1]

            #with open(cas_file, 'r') as file:
                #cas = pd.read_csv(file, header=0)
            
            #d['concept_cas'] = cas['concept_cas'].iloc[-1]
            #d['task_cas'] = cas['task_cas'].iloc[-1]

            performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
        except:
            pass




########## Task & Concept Accuracy Plot ##########

def get_df_name(df):
    if df=='xor':
        return 'XOR'
    elif df=='dot':
        return 'DOT'
    elif df=='trigonometry':
        return 'Trigonometry'
    elif df=='and':
        return 'AND'
    elif df=='or':
        return 'OR'
    elif df=='cebab':
        return 'CEBaB'
    elif df=='mnist_addition':
        return 'MNIST+'
    elif df=='mnist_addition_incomplete':
        return 'MNIST-Add-Inc.'
    elif df=='cub':
        return 'CUB200'
    elif df=='imdb':
        return 'IMDB'
    elif df=='celeba':
        return 'CelebA'
    elif df=='mnist_even_odd':
        return 'MNIST-E/O'

#df = performance.copy()
# Filter data for 'task' and 'concept'
task_df = performance.copy()
task_df = task_df.rename(columns={'task': 'accuracy'})
concept_df = performance.copy()
concept_df = concept_df.rename(columns={'concept': 'accuracy'})

# Compute mean and std for 'task'
task_stats = task_df.groupby(['model', 'dataset']).agg(
    avg_accuracy_task=('accuracy', 'mean'),
    std_accuracy_task=('accuracy', 'std')
).reset_index().fillna(0)

# Compute mean and std for 'concept'
concept_stats = concept_df.groupby(['model', 'dataset']).agg(
    avg_accuracy_concept=('accuracy', 'mean'),
    std_accuracy_concept=('accuracy', 'std')
).reset_index().fillna(0)

# Merge the two DataFrames on 'model' and 'dataset'
merged_stats = pd.merge(task_stats, concept_stats, on=['model', 'dataset'])

# Define font properties
title_font = {'size': 24, 'weight': 'bold'}
label_font = {'size': 24}
tick_font = {'size': 10}

marker_size = 14
# Define a dictionary to associate marker, name, and color to each model
model_styles = {
    'v_cem': {'marker': 'D', 'name': 'V-CEM (Ours)', 'color': 'tab:green', 'size': marker_size},
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:purple', 'size': marker_size},
    'cbm_linear': {'marker': 's', 'name': 'CBM+Linear', 'color': 'tab:orange', 'size': marker_size},
    'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:red', 'size': marker_size},
    'blackbox': {'marker': 'o', 'name': 'Black-box', 'color': 'tab:blue', 'size': marker_size},
    'prob_cbm': {'marker': 'X', 'name': 'Prob-CBM', 'color': 'tab:cyan', 'size': marker_size}
}

# Define the custom order
custom_order = ['mnist_even_odd', 'mnist_addition' ,'celeba', 'cebab', 'imdb']

merged_stats = merged_stats.sort_values('dataset')
merged_stats['dataset'] = pd.Categorical(merged_stats['dataset'], categories=custom_order, ordered=True)

fig, axes = plt.subplots(1, len(merged_stats['dataset'].unique()), figsize=(15, 4), sharey=False, sharex=False)

for idx, dataset in enumerate(merged_stats['dataset'].unique()):
    ax = axes[idx]
    data = merged_stats[merged_stats['dataset'] == dataset]
    for model in data['model']:
        model_data = data[data['model'] == model]
        style = model_styles[model]
        ax.errorbar(model_data['avg_accuracy_concept'], model_data['avg_accuracy_task'],
                    xerr=model_data['std_accuracy_concept'], yerr=model_data['std_accuracy_task'],
                    fmt=style['marker'], label=style['name'], color=style['color'], markersize=style['size'],
                    markeredgewidth=0.5, markeredgecolor='black', alpha=0.7)
    ax.set_title(get_df_name(dataset), fontdict=title_font)
    ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
    ax.minorticks_off()
    ax.grid(True, zorder=0)
    #values_range = np.arange(0, 1.1, 0.1)
    #ax.set_yticks(values_range)  # Set y ticks from 0 to 1
    #ax.set_yticklabels([f'{x:.1f}' for x in values_range])  # Set y tick labels from 0 to 1
    #ax.set_xticks(values_range)  # Set x ticks from 0 to 1
    #ax.set_xticklabels([f'{x:.1f}' for x in values_range])  # Set x tick labels from 0 to 1
    if idx == 0:
        ax.set_ylabel('Task Acc', fontdict=label_font)
    ax.set_xlabel('Concept Acc', fontdict=label_font)

# Create custom legend handles
custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=style['size'], label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in model_styles.values()]

# Create a single legend below the plots
fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()#(rect=[0, 0.1, 1, 0.95])
#plt.savefig('figs/performance.pdf')

plt.show()



########## Task Accuracy Table ##########

task_avg = task_stats[['model', 'dataset', 'avg_accuracy_task']]
task_std = task_stats[['model', 'dataset', 'std_accuracy_task']]

# Merge task_avg and task_std dataframes on 'model' and 'dataset'
merged_task = pd.merge(task_avg, task_std, on=['model', 'dataset'])

# Create a pivot table with the desired format
pivot_table_avg = task_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_task'])
pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
pivot_table_std = task_std.pivot(index='model', columns='dataset', values=['std_accuracy_task'])
pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

final_table = pd.DataFrame()
for i, row in pivot_table_avg.iterrows():
    d={}
    for j in pivot_table_std.columns:
        acc = row[j]*100
        std = pivot_table_std.loc[i, j]*100
        d[j] = f"{acc:.2f} ± {std:.2f}"
    # add a column to the final_table dataframe called row.name which contains d
    final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
    
# Reindex the columns of final_table according to the custom order
final_table = final_table.reindex(columns=custom_order)

print(final_table)


########## Concept Accuracy Table ##########

task_avg = concept_stats[['model', 'dataset', 'avg_accuracy_concept']]
task_std = concept_stats[['model', 'dataset', 'std_accuracy_concept']]

# Merge task_avg and task_std dataframes on 'model' and 'dataset'
merged_task = pd.merge(task_avg, task_std, on=['model', 'dataset'])

# Create a pivot table with the desired format
pivot_table_avg = task_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_concept'])
pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
pivot_table_std = task_std.pivot(index='model', columns='dataset', values=['std_accuracy_concept'])
pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

final_table = pd.DataFrame()
for i, row in pivot_table_avg.iterrows():
    d={}
    for j in pivot_table_std.columns:
        acc = row[j]*100
        std = pivot_table_std.loc[i, j]*100
        d[j] = f"{acc:.2f} ± {std:.2f}"
    # add a column to the final_table dataframe called row.name which contains d
    final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
    
# Reindex the columns of final_table according to the custom order
final_table = final_table.reindex(columns=custom_order)

print(final_table)


########## Intervention results ##########

performance = pd.DataFrame()

for exp in exps_path:
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/interventions.csv')        
    if os.path.exists(conf_file) and os.path.exists(result_file):
        with open(result_file, 'r') as file:
            d = pd.read_csv(result_file)[['noise','p_int','f1','accuracy']]
        
        with open(conf_file, 'r') as file:
            conf = yaml.safe_load(file)
        d['seed'] = conf['seed']
        d['dataset'] = conf['dataset']['metadata']['name']
        d['model'] = conf['model']['metadata']['name']

        performance = pd.concat([performance, d], ignore_index=True)


########## Intervention OOD results ########## 

def plot_intervention_results(df, metric='accuracy', title_font=None, label_font=None, tick_font=None, legend_font=None):
    unique_noises = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    unique_datasets = custom_order
    n_cols = len(unique_noises)
    n_rows = len(unique_datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey='row')
    
    for i, dataset in enumerate(unique_datasets):
        for j, noise in enumerate(unique_noises):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            grouped_data = data.groupby(['p_int', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            for model in grouped_data['model'].unique():
                model_data = grouped_data[grouped_data['model'] == model]
                style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                #ax.errorbar(model_data['p_int'], model_data['mean_metric'], yerr=model_data['std_metric'],
                #            fmt=style['marker'], color=style['color'], markersize=style['size'], label=style['name'])
                ax.plot(model_data['p_int'], model_data['mean_metric'], color=style['color'], linestyle='-', alpha=0.5)
                ax.scatter(model_data['p_int'], model_data['mean_metric'], marker=style['marker'], color=style['color'], s=style['size']**2, label=style['name'], edgecolor='black', alpha=0.5)
                ax.fill_between(model_data['p_int'], model_data['mean_metric'] - model_data['std_metric'], model_data['mean_metric'] + model_data['std_metric'], color=style['color'], alpha=0.2)
            if i == 0:
                ax.set_title(r'$\theta$'+f'={noise}', fontsize=title_font['size'])
            if i == n_rows - 1:
                ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
            if j == 0:
                ax.set_ylabel(f'{get_df_name(dataset)}', fontsize=label_font['size'])
            ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
            ax.minorticks_off()
            ax.grid(True)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Create a single legend below the plots
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    for handle in handles:
        handle.set_alpha(1)  # Remove transparency from legend markers

    # Create custom legend handles
    custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=style['size']+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in model_styles.values()]

    # Create a single legend below the plots
    fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plt.savefig('figs/intervention.pdf')
    plt.show()

# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 44, 'weight': 'bold'}
label_font = {'size': 44}
tick_font = {'size': 30}
plot_intervention_results(performance, metric='accuracy', title_font=title_font, label_font=label_font, tick_font=tick_font, legend_font=legend_font)



########## Intervention ID results ##########

def plot_intervention_results(df, metric='accuracy', title_font=None, label_font=None, tick_font=None, legend_font=None):
    unique_noises = [0]
    unique_datasets = custom_order
    n_cols = len(unique_noises)
    n_rows = len(unique_datasets)
    fig, axes = plt.subplots(n_cols, n_rows, figsize=(30, 7), sharex=True, sharey=True)
    
    for i, dataset in enumerate(unique_datasets):
        for j, noise in enumerate(unique_noises):
            ax = axes[i] #axes[i, j] if n_rows > 1 else axes[j]
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            grouped_data = data.groupby(['p_int', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            for model in grouped_data['model'].unique():
                model_data = grouped_data[grouped_data['model'] == model]
                style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                #ax.errorbar(model_data['p_int'], model_data['mean_metric'], yerr=model_data['std_metric'],
                #            fmt=style['marker'], color=style['color'], markersize=style['size'], label=style['name'])
                ax.plot(model_data['p_int'], model_data['mean_metric'], color=style['color'], linestyle='-', alpha=0.5)
                ax.scatter(model_data['p_int'], model_data['mean_metric'], marker=style['marker'], color=style['color'], s=style['size']**2, label=style['name'], edgecolor='black', alpha=0.5)
                ax.fill_between(model_data['p_int'], model_data['mean_metric'] - model_data['std_metric'], model_data['mean_metric'] + model_data['std_metric'], color=style['color'], alpha=0.2)
            ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
            if j == 0:
                ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])
            if i == 0:
                ax.set_ylabel('Task Acc', fontsize=label_font['size'])
            ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
            ax.minorticks_off()
            ax.grid(True)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Create a single legend below the plots
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    for handle in handles:
        handle.set_alpha(1)  # Remove transparency from legend markers

    # Create custom legend handles
    custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=style['size']+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in model_styles.values()]

    # Create a single legend below the plots
    fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.2))

    plt.tight_layout()
    plt.savefig('figs/intervention_id.pdf')
    plt.show()

# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 44, 'weight': 'bold'}
label_font = {'size': 44}
tick_font = {'size': 30}
plot_intervention_results(performance, metric='accuracy', title_font=title_font, label_font=label_font, tick_font=tick_font, legend_font=legend_font)


######### Concept Representation Cohesiveness (CRC) ##########

import torch
from sklearn.metrics import silhouette_score
from tqdm import tqdm

performance = pd.DataFrame()

for exp in tqdm(exps_path):
    d = {}
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    #result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/metrics.csv')  
    concept_path = os.path.join(exp, 'logs/experiment_metrics/version_0/concept_prediction.pt')
    latent_path = os.path.join(exp, 'logs/experiment_metrics/version_0/latents.pt')

    ##### ELIMINATE THE ADDITONAL AND OPERATION
    if os.path.exists(conf_file) and os.path.exists(concept_path) and os.path.exists(latent_path):
        with open(conf_file, 'r') as file:
            conf = yaml.safe_load(file)

        d['seed'] = conf['seed']
        d['dataset'] = conf['dataset']['metadata']['name']
        d['model'] = conf['model']['metadata']['name']
        d['path'] = exp

        latent_tensor = torch.load(latent_path).numpy()
        concept_tensor = torch.load(concept_path).numpy()

        silhouette_scores = []

        for i in range(concept_tensor.shape[1]):
            labels = concept_tensor[:, i]
            if d['model'] in ['v_cem', 'cem', 'prob_cbm']:
                latent = latent_tensor.reshape(-1, concept_tensor.shape[1], 16)[:,i,:]
            else:
                latent = latent_tensor[:, i].reshape(-1,1)
            score = silhouette_score(latent, labels, metric='l1')
            silhouette_scores.append(score)

        mean_silhouette = sum(silhouette_scores)/len(silhouette_scores)
        d['silhouette'] = mean_silhouette
        performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)

performance.dropna(inplace=True)

silhouette_stats = performance.groupby(['dataset', 'model']).agg(
    avg_silhouette=('silhouette', 'mean'),
    std_silhouette=('silhouette', 'std')
).reset_index()

print(silhouette_stats)

task_avg = silhouette_stats[['model', 'dataset', 'avg_silhouette']]
task_std = silhouette_stats[['model', 'dataset', 'std_silhouette']]

# Merge task_avg and task_std dataframes on 'model' and 'dataset'
merged_task = pd.merge(task_avg, task_std, on=['model', 'dataset'])

# Create a pivot table with the desired format
pivot_table_avg = task_avg.pivot(index='model', columns='dataset', values=['avg_silhouette'])
pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
pivot_table_std = task_std.pivot(index='model', columns='dataset', values=['std_silhouette'])
pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

final_table = pd.DataFrame()
for i, row in pivot_table_avg.iterrows():
    d={}
    for j in pivot_table_std.columns:
        acc = row[j]
        std = pivot_table_std.loc[i, j]
        d[j] = f"{acc:.2f} ± {std:.2f}"
    # add a column to the final_table dataframe called row.name which contains d
    final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
    
# Reindex the columns of final_table according to the custom order
final_table = final_table.reindex(columns=custom_order)

print(final_table)


########## Concept Space visualization ##########

# select only the rows associated to cebab and v_cem, cem

selected_datasets = ['cebab', 'mnist_even_odd', 'mnist_addition', 'celeba', 'imdb']

for selected_dataset in selected_datasets:
    filtered = performance[(performance['dataset'] == selected_dataset) & (performance['model'].isin(['cem', 'v_cem', 'prob_cbm'])) & (performance['seed'] == 1)]

    # load concept latent space
    v_cem_path = filtered[filtered['model']=='v_cem'].iloc[0]['path']
    cem_path = filtered[filtered['model']=='cem'].iloc[0]['path']
    prob_cbm = filtered[filtered['model']=='prob_cbm'].iloc[0]['path']

    # the true tensor is the same so we can either take it form cem or v_cem path without problems
    cem_concept_path = os.path.join(cem_path, 'logs/experiment_metrics/version_0/concept_prediction.pt')
    v_cem_concept_path = os.path.join(v_cem_path, 'logs/experiment_metrics/version_0/concept_prediction.pt')
    prob_cbm_concept_path = os.path.join(prob_cbm, 'logs/experiment_metrics/version_0/concept_prediction.pt')

    cem_latent_path = os.path.join(cem_path, 'logs/experiment_metrics/version_0/latents.pt')
    v_cem_latent_path = os.path.join(v_cem_path, 'logs/experiment_metrics/version_0/latents.pt')
    prob_cbm_latent_path = os.path.join(prob_cbm, 'logs/experiment_metrics/version_0/latents.pt')

    cem_latent_tensor = torch.load(cem_latent_path).numpy()
    v_cem_latent_tensor = torch.load(v_cem_latent_path).numpy()
    cem_concept_tensor = torch.load(cem_concept_path).numpy()
    v_cem_concept_tensor = torch.load(v_cem_concept_path).numpy()
    prob_cbm_latent_tensor = torch.load(prob_cbm_latent_path).numpy()
    prob_cbm_concept_tensor = torch.load(prob_cbm_concept_path).numpy()

    # load the concept "expetect values" for both concept states
    # Load the model checkpoint

    def get_ckpt_path(folder):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.ckpt'):
                    return os.path.join(root, file)
        return None

    model_ckpt_path = os.path.join(get_ckpt_path(v_cem_path))
    model = torch.load(model_ckpt_path)
    pos_embs = model['state_dict']['prototype_emb_pos']
    neg_embs = model['state_dict']['prototype_emb_neg']


    #from sklearn.manifold import TSNE
    from openTSNE import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from tqdm import tqdm

    # Define font properties
    title_font = {'size': 36, 'weight': 'bold'}
    label_font = {'size': 36}
    tick_font = {'size': 26}
    legend_font = {'size': 34}

    concept_tensors = [cem_concept_tensor, prob_cbm_concept_tensor, v_cem_concept_tensor]

    fig, axes = plt.subplots(3, concept_tensors[0].shape[1], figsize=(20, 10), sharex=True, sharey=True)

    if selected_dataset=='cebab':
        concept_names = ['Food', 'Ambiance', 'Service', 'Noise']
    elif selected_dataset=='imdb':
        concept_names = ["Acting", "Storyline", "Emotional Arousal", "Cinematography"]
    elif selected_dataset=='celeba':
        concept_names = [x.replace('_',' ') for x in ["No_Beard", "Young", "Attractive", "Mouth_Slightly_Open", "Smiling", "Wearing_Lipstick", "High_Cheekbones"]]
    elif selected_dataset in ['mnist_even_odd', 'mnist_addition']:
        concept_names = [f"Number {i+1}" for i in range(concept_tensors[0].shape[1])]
    else:
        concept_names = [f"Concept {i+1}" for i in range(concept_tensors[0].shape[1])]

    for row, latent_tensor in enumerate([cem_latent_tensor, prob_cbm_latent_tensor, v_cem_latent_tensor]):
        concept_tensor = concept_tensors[row]
        for i in tqdm(range(concept_tensor.shape[1])):
            # Perform t-SNE on the latent tensor
            tsne = TSNE().fit(latent_tensor.reshape(-1, concept_tensor.shape[1], 16)[:,i,:])
            latent_2d = tsne.transform(latent_tensor.reshape(-1, concept_tensor.shape[1], 16)[:,i,:])
            
            # Define colors based on cluster labels
            colors = ['tab:green' if label == 1 else 'tab:red' for label in concept_tensor[:, i]]

            # Plot the t-SNE results
            scatter = axes[row, i].scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, alpha=0.6)
            if row == 0:
                axes[row, i].set_title(concept_names[i], fontdict=title_font)
            if row == 1:
                axes[row, i].set_xlabel('', fontdict=label_font)
            if i == 0:
                if row==0:
                    axes[row, i].set_ylabel('CEM', fontdict=label_font)
                elif row==1:
                    axes[row, i].set_ylabel('Prob-CBM', fontdict=label_font)
                else:
                    axes[row, i].set_ylabel('V-CEM', fontdict=label_font)

            axes[row, i].tick_params(axis='both', which='major', labelsize=tick_font['size'])
            axes[row, i].grid(True)
            axes[row, i].minorticks_off()

            if row == 2:
                # Plot the positive and negative prototype embeddings
                pos_emb_2d = tsne.transform(pos_embs[i].reshape(1, -1).cpu())
                neg_emb_2d = tsne.transform(neg_embs[i].reshape(1, -1).cpu())
                axes[row, i].scatter(pos_emb_2d[:, 0], pos_emb_2d[:, 1], marker='X', s=400, label='$\mu_j^+$', 
                                                                        facecolor='tab:green', edgecolors='black', linewidths=2)
                axes[row, i].scatter(neg_emb_2d[:, 0], neg_emb_2d[:, 1], marker='X', s=400, label='$\mu_j^-$', 
                                                                        facecolor='tab:red', edgecolors='black', linewidths=2)

    # Create custom legend handles
    custom_handles = [Line2D([0], [0], marker='X', markersize=24, label='$\mu_j^+$', 
                            markerfacecolor='tab:green', markeredgecolor='black', linewidth=2, linestyle='None'),
                    Line2D([0], [0], marker='X', markersize=24, label='$\mu_j^-$', 
                            markerfacecolor='tab:red', markeredgecolor='black', linewidth=2, linestyle='None')]

    # Create a single legend below the plots
    fig.legend(handles=custom_handles, loc='lower center', ncol=2, fontsize=legend_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    plt.savefig(f"figs/latent_space{selected_dataset}.pdf")
    plt.show()


########### Ablation Study ##########

path = '/home/fdesantis/Adversary-Aware-Concept-Embedding-Model/multirun/2025-02-27/10-18-23'
exps = os.listdir(path)
exps_path = [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

performance = pd.DataFrame()

for exp in exps_path:
    d = {}
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/metrics.csv')  
    #cas_file = os.path.join(exp, 'logs/experiment_metrics/version_0/cas.csv')   
    print(exp)
    if os.path.exists(conf_file) and os.path.exists(result_file):
        with open(conf_file, 'r') as file:
            conf = yaml.safe_load(file)
        d['seed'] = conf['seed']
        d['dataset'] = conf['dataset']['metadata']['name']
        d['model'] = conf['model']['metadata']['name']
        d['prior_penalty'] = conf['kl_penalty']

        with open(result_file, 'r') as file:
            result = pd.read_csv(file, header=0)

        d['task'] = result['test_task_acc'].iloc[-1]
        d['concept'] = result['test_concept_acc'].iloc[-1]

        performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)

performance['dataset'] = performance.apply(lambda x: get_df_name(x['dataset']), axis=1)


# Assuming df is your DataFrame
# Group by dataset and prior_penalty, then compute mean and std dev of task
grouped = performance.groupby(['dataset', 'prior_penalty']).agg({'task': ['mean', 'std']}).reset_index()

# Flatten the MultiIndex columns
grouped.columns = ['dataset', 'prior_penalty', 'task_mean', 'task_std']

# Plotting
fig, ax = plt.subplots(figsize=(5, 4))

# Iterate over each dataset and plot
for dataset in grouped['dataset'].unique():
    data = grouped[grouped['dataset'] == dataset]
    ax.errorbar(data['prior_penalty'], data['task_mean'], yerr=data['task_std'], label=dataset, capsize=5, linestyle='-', marker='o')

ax.set_xlabel('$\lambda_p$', fontsize=18)
ax.set_ylabel('Accuracy', fontsize=18)
ax.set_xscale('log')
ax.grid(True)
ax.minorticks_off()

# Show each x tick which has a corresponding point in the figure
ax.set_xticks(grouped['prior_penalty'].unique())
tick_labels = ['1e-4', '1e-3', '1e-2', '5e-2', '1e-1', '1', '10']
ax.set_xticklabels(tick_labels)

# Remove the title
ax.set_title('')

# Set font sizes for ticks
ax.tick_params(axis='both', which='major', labelsize=8)

# Create custom legend handles
handles, labels = ax.get_legend_handles_labels()
colors = ['black', 'red']
custom_handles = [plt.Line2D([0], [1], color=colors[i],  label=label) for i, (handle, label) in enumerate(zip(handles, labels))]

# Create a single legend in the upper right corner
ax.legend(handles=custom_handles, title='Dataset', loc='lower left', frameon=True, fontsize=12, title_fontsize=14)  # Change fontsize here
plt.tight_layout()
plt.savefig('figs/prior_penalty.pdf')
plt.show()


performance = pd.DataFrame()

for exp in exps_path:
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/interventions.csv')        
    if os.path.exists(conf_file) and os.path.exists(result_file):
        with open(result_file, 'r') as file:
            d = pd.read_csv(result_file)[['noise','p_int','f1','accuracy']]
        
        with open(conf_file, 'r') as file:
            conf = yaml.safe_load(file)
        d['seed'] = conf['seed']
        d['dataset'] = conf['dataset']['metadata']['name']
        d['model'] = conf['model']['metadata']['name']
        d['prior_penalty'] = conf['kl_penalty']

        performance = pd.concat([performance, d], ignore_index=True)

marker_size = 14
model_styles = {
    'v_cem': {'marker': 'x', 'name': 'V-CEM (Ours)', 'color': 'tab:green', 'size': marker_size},
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:purple', 'size': marker_size},
    'cbm_linear': {'marker': 's', 'name': 'CBM+Linear', 'color': 'tab:orange', 'size': marker_size},
    'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:red', 'size': marker_size},
    'blackbox': {'marker': 'o', 'name': 'Black-box', 'color': 'tab:blue', 'size': marker_size}
}
custom_order = ['mnist_even_odd', 'mnist_addition' ,'celeba', 'cebab', 'imdb']


from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import numpy as np

def plot_intervention_results(df, metric='accuracy', title_font=None, label_font=None, tick_font=None, legend_font=None):
    unique_noises = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    unique_datasets = df['dataset'].unique()
    unique_prior_penalties = sorted(df['prior_penalty'].unique())
    n_cols = len(unique_noises)
    n_rows = len(unique_datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey='row')
    
    # Define a color map from yellow to blue and reverse it
    colors = cm.plasma(np.linspace(0, 1, len(unique_prior_penalties)))[::-1]
    penalty_color_map = {penalty: color for penalty, color in zip(unique_prior_penalties, colors)}
    
    for i, dataset in enumerate(unique_datasets):
        for j, noise in enumerate(unique_noises):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            grouped_data = data.groupby(['p_int', 'model', 'prior_penalty']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            for prior_penalty in unique_prior_penalties:
                for model in grouped_data['model'].unique():
                    model_data = grouped_data[(grouped_data['model'] == model) & (grouped_data['prior_penalty'] == prior_penalty)]
                    style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                    color = penalty_color_map[prior_penalty]
                    ax.errorbar(model_data['p_int'], model_data['mean_metric'], yerr=model_data['std_metric'],
                                fmt=style['marker'], color=color, markersize=style['size'], label=f"{style['name']} (Penalty={prior_penalty})")
                    ax.plot(model_data['p_int'], model_data['mean_metric'], color=color, linestyle='-', alpha=0.5)
                    ax.scatter(model_data['p_int'], model_data['mean_metric'], marker=style['marker'], color=color, s=style['size']**2, label=f"{style['name']} (Penalty={prior_penalty})", edgecolor='black', alpha=0.5)
            if i == 0:
                ax.set_title(r'$\theta$'+f'={noise}', fontsize=title_font['size'])
            if i == n_rows - 1:
                ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
            if j == 0:
                ax.set_ylabel(f'{get_df_name(dataset)}', fontsize=label_font['size'])
            ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
            ax.minorticks_off()
            ax.grid(True)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Create a single legend below the plots
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # Create custom legend handles for prior penalties
    custom_handles = [plt.Line2D([0], [0], color=color, lw=4, label=f'$\lambda_p$={penalty}') for penalty, color in penalty_color_map.items()]

    fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles)//2, fontsize=legend_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.5))

    plt.tight_layout()
    plt.savefig('figs/prior_ablation_int.pdf')
    plt.show()

# Call the function with the desired metric and font properties
legend_font = {'size': 42}
title_font = {'size': 42, 'weight': 'bold'}
label_font = {'size': 42}
tick_font = {'size': 30}
plot_intervention_results(performance, metric='accuracy', title_font=title_font, label_font=label_font, tick_font=tick_font, legend_font=legend_font)
