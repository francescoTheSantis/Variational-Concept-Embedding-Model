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
        target_label = self.target_labels[index]
        return data, concept_label, target_label


class CelebADataset(CelebA):
    """
    The CelebA dataset is a large-scale face attributes dataset with more than
    200K celebrity images, each with 40 attribute annotations. This class
    extends the CelebA dataset to extract concept and task attributes based on
    class attributes.

    The dataset can be downloaded from the official
    website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.

    Attributes:
        root: The root directory where the dataset is stored.
        split: The split of the dataset to use. Default is 'train'.
        transform: The transformations to apply to the images. Default is None.
        download: Whether to download the dataset if it does not exist. Default
            is False.
        class_attributes: The class attributes to use for the task. Default is
            None.
    """
    def __init__(
        self, root: str, split: str = 'train',
        transform = None,
        download: bool = False,
        class_attributes: List[str] = None,
    ):
        super(CelebADataset, self).__init__(
            root,
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )

        # Set the class attributes
        if class_attributes is None:
            # Default to 'Attractive' if no class_attributes provided
            self.class_idx = [self.attr_names.index('Attractive')]
        else:
            # Use the provided class attributes
            self.class_idx = [
                self.attr_names.index(attr) for attr in class_attributes
            ]

        self.attr_names = [string for string in self.attr_names if string]

        # Determine concept and task attribute names based on class attributes
        self.concept_attr_names = [
            attr for i, attr in enumerate(self.attr_names)
            if i not in self.class_idx
        ]
        self.task_attr_names = [self.attr_names[i] for i in self.class_idx]

    def __getitem__(self, index: int):
        image, attributes = super(CelebADataset, self).__getitem__(index)

        # Extract the target (y) based on the class index
        y = torch.stack([attributes[i] for i in self.class_idx])

        # Extract concept attributes, excluding the class attributes
        concept_attributes = torch.stack([
            attributes[i] for i in range(len(attributes))
            if i not in self.class_idx
        ])

        return image, concept_attributes, y



# create a class that, given the specific cusotm dataset above, generates the train,val and test splits (batched)
# and returns the dataloaders for each split
class DataLoader:
    def __init__(self, dataset, batch_size, train_size, val_size, test_size, random_state=0):
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




