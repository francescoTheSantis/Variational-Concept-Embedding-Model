from torchvision.datasets import CelebA
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset

def _boolean_op(size, operator, random_state=42):
    # sample from uniform distribution
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, (size, 2))
    c = np.stack([
        x[:, 0] > 0.5,
        x[:, 1] > 0.5,
    ]).T

    if operator == 'xor':
        y = np.logical_xor(c[:, 0], c[:, 1])
    elif operator == 'and':
        y = np.logical_and(c[:, 0], c[:, 1])
    elif operator == 'or':
        y = np.logical_or(c[:, 0], c[:, 1])

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)
    return x, c, y.unsqueeze(-1), None, ['C1', 'C2'], [operator]


def _trigonometry(size, random_state=42):
    np.random.seed(random_state)
    h = np.random.normal(0, 2, (size, 3))
    x, y, z = h[:, 0], h[:, 1], h[:, 2]

    # raw features
    input_features = np.stack([
        np.sin(x) + x,
        np.cos(x) + x,
        np.sin(y) + y,
        np.cos(y) + y,
        np.sin(z) + z,
        np.cos(z) + z,
        x ** 2 + y ** 2 + z ** 2,
    ]).T

    # concepts
    concepts = np.stack([
        x > 0,
        y > 0,
        z > 0,
    ]).T

    # task
    downstream_task = (x + y + z) > 1

    input_features = torch.FloatTensor(input_features)
    concepts = torch.FloatTensor(concepts)
    downstream_task = torch.FloatTensor(downstream_task)
    return (
        input_features,
        concepts,
        downstream_task.unsqueeze(-1),
        None,
        ['C1', 'C2', 'C3'],
        ['sumGreaterThan1'],
    )

def _dot(size, random_state=42):
    np.random.seed(random_state)
    # sample from normal distribution
    emb_size = 2
    # Generate the latent vectors
    v1 = np.random.randn(size, emb_size) * 2
    v2 = np.ones(emb_size)
    v3 = np.random.randn(size, emb_size) * 2
    v4 = -np.ones(emb_size)
    # Generate the sample
    x = np.hstack([v1+v3, v1-v3])
    
    # Now the concept vector
    c = np.stack([
        np.dot(v1, v2).ravel() > 0,
        np.dot(v3, v4).ravel() > 0,
    ]).T
    # And finally the label
    y = ((v1*v3).sum(axis=-1) > 0).astype(np.int64)

    # We NEED to put all of these into torch Tensors (THIS IS VERY IMPORTANT)
    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.Tensor(y)
    return (x, c, y, None, ['dotV1V2GreaterThan0', 'dotV3V4GreaterThan0'], ['dotV1V3GreaterThan0'])

class ToyDataset(Dataset):
    def __init__(self, dataset: str, size: int, random_state: int = 42):
        self.size = size
        self.random_state = random_state
        (
            self.data,
            self.concept_labels,
            self.target_labels,
            self.dag,
            self.concept_attr_names,
            self.task_attr_names
        ) = self._load_data(dataset)

    def _load_data(self, dataset):
        if dataset in ['xor','and','or']:
            return _boolean_op(self.size, dataset, self.random_state)
        elif dataset == 'trigonometry':
            return _trigonometry(self.size, self.random_state)
        elif dataset == 'dot':
            return _dot(self.size, self.random_state)
        else:
            raise ValueError(f"Unknown dataset '{dataset}'")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = self.data[index]
        concept_label = self.concept_labels[index]
        target_label = self.target_labels[index].squeeze()
        return data.float(), concept_label.float(), target_label.float()

# create a class that, given the specific cusotm dataset above, generates the train,val and test splits (batched)
# and returns the dataloaders for each split
class Toy_DataLoader:
    def __init__(self, dataset, batch_size, train_size, val_size, test_size, random_state=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def get_data_loaders(self):
        # create the dataset
        if self.dataset == 'celeba':
            train_dataset = CelebADataset(
                root='data', split='train', download=True,
                class_attributes=['Attractive']
            )
            train_dataset = CelebADataset(
                root='data', split='valid', download=True,
                class_attributes=['Attractive']
            )
            train_dataset = CelebADataset(
                root='data', split='test', download=True,
                class_attributes=['Attractive']
            )
        else:
            dataset = ToyDataset(self.dataset, self.train_size + self.val_size + self.test_size, self.random_state)
            # split the dataset
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [self.train_size, self.val_size, self.test_size],
                generator=torch.Generator().manual_seed(self.random_state)
            )
        # create the dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    

def toy_loader_function(dataset, batch_size, train_size, val_size, test_size, random_state=42):
    loaded_train, loaded_val, loaded_test = Toy_DataLoader(dataset, batch_size, train_size, val_size, test_size, random_state).get_data_loaders()
    # Normalize the first element in the batch of all the splits
    loaded_train = normalize(loaded_train)
    loaded_val = normalize(loaded_val)
    loaded_test = normalize(loaded_test)

    return loaded_train, loaded_val, loaded_test

def normalize(loader):
    x_list, c_list, y_list = [], [], []
    for x, c, y in loader:
        x_list.append(x)
        c_list.append(c)
        y_list.append(y)
    x = torch.cat(x_list, dim=0)
    c = torch.cat(c_list, dim=0)
    y = torch.cat(y_list, dim=0)

    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x = (x - x_mean) / x_std
    # recreate the loader
    _loader = torch.utils.data.TensorDataset(x, c, y)
    loader = torch.utils.data.DataLoader(_loader, batch_size=loader.batch_size, shuffle=False)
    return loader




