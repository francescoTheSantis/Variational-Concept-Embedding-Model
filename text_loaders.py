import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
def process(elem):
    if elem in ['Negative','unknown']:
        return 0
    else:
        return 1

def process2(elem):
    if elem == 'Positive':
        return 1
    else:
        return 0 

class CEBABDataset(Dataset):
    def __init__(self, split, model_name='all-MiniLM-L6-v2'):

        self.data = pd.read_csv(f'data/cebab/cebab_{split}.csv')
        self.data['food'] = self.data.apply(lambda row: process(row['food']), axis=1)
        self.data['ambiance'] = self.data.apply(lambda row: process(row['ambiance']), axis=1)
        self.data['service'] = self.data.apply(lambda row: process(row['service']), axis=1)
        self.data['noise'] = self.data.apply(lambda row: process(row['noise']), axis=1)
        self.data['bin_rating'] = self.data.apply(lambda row: process2(row['bin_rating']), axis=1)
        self.model = SentenceTransformer(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract review, concept annotations, and label
        review = self.data.loc[idx, 'review']  # Column name for the review text
        text_embedding = self.model.encode(review, convert_to_tensor=True)
        concepts = self.data.loc[idx, ['food', 'service', 'ambiance', 'noise']].values

        label = self.data.loc[idx, 'bin_rating']  # Column name for the label

        # Convert concepts to numeric and handle invalid entries
        concepts = np.array(concepts, dtype=np.float32)  # Ensure all are numeric
        label = int(label)  # Ensure label is an integer

        # Convert to tensors
        concepts_tensor = torch.tensor(concepts, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return text_embedding, concepts_tensor, label_tensor



def collate_fn(batch):
    """
    Custom collate function for batching data into a format suitable for training.

    Args:
        batch (list): List of data samples.

    Returns:
        Tuple: Batched tensor of text embeddings, concept annotations, and labels.
    """
    text_embeddings = torch.stack([item[0] for item in batch])
    concepts = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])

    return text_embeddings, concepts, labels


class IMDBDataset(Dataset):
    def __init__(self, split, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the dataset with a CSV file and tokenizer.

        Args:
            csv_file (str): Path to the CSV file (train/val/test split).
            tokenizer_model (str): Hugging Face model for Siamese embeddings.
            max_length (int): Maximum sequence length for tokenization.
        """

        self.folder = 'data/imdb'
        self.data = pd.concat([pd.read_csv(f'{self.folder}/IMDB-{split}-generated.csv'), pd.read_csv(f'{self.folder}/IMDB-{split}-manual.csv')]).reset_index()
        self.data['acting'] = self.data.apply(lambda row: process(row['acting']), axis=1)
        self.data['storyline'] = self.data.apply(lambda row: process(row['storyline']), axis=1)
        self.data['emotional arousal'] = self.data.apply(lambda row: process(row['emotional arousal']), axis=1)
        self.data['cinematography'] = self.data.apply(lambda row: process(row['cinematography']), axis=1)
        self.data['soundtrack'] = self.data.apply(lambda row: process(row['soundtrack']), axis=1)
        self.data['directing'] = self.data.apply(lambda row: process(row['directing']), axis=1)
        self.data['background setting'] = self.data.apply(lambda row: process(row['background setting']), axis=1)
        self.data['editing'] = self.data.apply(lambda row: process(row['editing']), axis=1)
        self.data['sentiment'] = self.data.apply(lambda row: process2(row['sentiment']), axis=1)
        self.model = SentenceTransformer(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract review, concept annotations, and label
        review = self.data.loc[idx, 'review']
        text_embedding = self.model.encode(review, convert_to_tensor=True)
        concepts = self.data.loc[idx, ['acting', 'storyline', 'emotional arousal', 'cinematography', 'soundtrack', 'directing', 'background setting', 'editing']].values

        label = self.data.loc[idx, 'sentiment']  # Column name for the label

        # Convert concepts to numeric and handle invalid entries
        concepts = np.array(concepts, dtype=np.float32)  # Ensure all are numeric
        label = int(label)  # Ensure label is an integer

        # Convert to tensors
        concepts_tensor = torch.tensor(concepts, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return text_embedding, concepts_tensor, label_tensor

