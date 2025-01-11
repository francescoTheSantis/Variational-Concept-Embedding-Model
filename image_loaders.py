import torch
from torchvision.datasets import CelebA
from typing import List
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch import nn
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, mean, std, images, labels, digits):
        
        tensor = [] #np.zeros((len(images), 28, 56)) #torch.Tensor(len(images), 28, 56)
        for i in range(len(images)):
            tmp = torch.Tensor(images[i]).numpy()
            tmp = np.clip(tmp * 255, 0, 255).astype(np.uint8)
            tmp = Image.fromarray(tmp)
            tensor.append(tmp)
            
        self.images = tensor               
        self.labels = labels
        self.mean = mean
        self.std = std
        self.digits = digits
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        label = self.labels[idx]
        image = self.transform(image)   

        digits = np.zeros(10, dtype=float)
        digits[self.digits[idx]] = 1 
        return image, digits, label
    

def MNIST_addition_loader(batch_size, val_size=0.1, seed=42, num_workers=3, pin_memory=True, shuffle=True):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    
    # fix the seed for both pytorch generator and numpy.random
    generator = torch.Generator().manual_seed(seed) 
    np.random.seed(seed)

    train_dataset = datasets.MNIST(root='./datasets/', train=True, download=True)
    test_dataset = datasets.MNIST(root='./datasets/', train=False)

    # Create composed training-set
    unique_pairs = [str(x)+str(y) for x in range(10) for y in range(10)]
    X_train = []
    y_train = []
    c_train = []
    y_train_lab = np.array([x[1] for x in train_dataset])
    y_test_lab = np.array([x[1] for x in test_dataset])
    y_digits = np.array([x[1] for x in test_dataset])
    samples_per_permutation = 1000
    for train_set_pair in unique_pairs:
        for _ in range(samples_per_permutation):
            rand_i = np.random.choice(np.where(y_train_lab == int(train_set_pair[0]))[0])
            rand_j = np.random.choice(np.where(y_train_lab == int(train_set_pair[1]))[0])
            temp_image = np.zeros((28,56), dtype="uint8")
            temp_image[:,:28] = train_dataset[rand_i][0]
            temp_image[:,28:] = train_dataset[rand_j][0]
            X_train.append(temp_image)
            y_train.append(y_train_lab[rand_i] + y_train_lab[rand_j])
            c_train.append([y_train_lab[rand_i], y_train_lab[rand_j]])  
    
    # Create composed test-set
    X_test = []
    y_test = []
    c_test = []
    samples_per_permutation = 100
    for test_set_pair in unique_pairs:
        for _ in range(samples_per_permutation):
            rand_i = np.random.choice(np.where(y_test_lab == int(test_set_pair[0]))[0])
            rand_j = np.random.choice(np.where(y_test_lab == int(test_set_pair[1]))[0])
            temp_image = np.zeros((28,56), dtype="uint8")
            temp_image[:,:28] = test_dataset[rand_i][0]
            temp_image[:,28:] = test_dataset[rand_j][0]
            X_test.append(temp_image)
            y_test.append(y_test_lab[rand_i] + y_test_lab[rand_j])
            c_test.append([y_test_lab[rand_i], y_test_lab[rand_j]])
    
    train_dataset = CustomDataset(mean, std, X_train, y_train, c_train)        
    test_dataset = CustomDataset(mean, std, X_test, y_test, c_test)        
    val_size = int(len(train_dataset) * val_size)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


class EmbeddingExtractor:
    def __init__(self, train_loader, val_loader, test_loader, device='cuda'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Load ResNet18 model pre-trained on ImageNet
        self.model = resnet18(pretrained=True)
        # Remove the fully connected layer to get embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _extract_embeddings(self, loader):
        """Helper function to extract embeddings for a given DataLoader."""
        embeddings = []
        concepts_list = []
        labels = []

        with torch.no_grad():
            for images, concepts, targets in loader:
                images = images.to(self.device)
                # Extract embeddings
                output = self.model(images)
                # Flatten the output from (batch_size, 512, 1, 1) to (batch_size, 512)
                output = output.view(output.size(0), -1)
                embeddings.append(output.cpu())
                concepts_list.append(concepts.cpu())
                labels.append(targets.cpu())

        # Concatenate all embeddings and labels
        embeddings = torch.cat(embeddings, dim=0)
        concepts = torch.cat(concepts_list, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, concepts.float(), labels

    def _create_loader(self, embeddings, concepts, labels, batch_size):
        """Helper function to create a DataLoader from embeddings and labels."""
        dataset = TensorDataset(embeddings, concepts, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def produce_loaders(self):
        """Produces new DataLoaders with embeddings instead of raw images."""
        train_embeddings, train_concepts, train_labels = self._extract_embeddings(self.train_loader)
        val_embeddings, val_concepts, val_labels = self._extract_embeddings(self.val_loader)
        test_embeddings, test_concepts, test_labels = self._extract_embeddings(self.test_loader)

        batch_size = self.train_loader.batch_size

        train_embeddings, train_concepts, train_labels = self._extract_embeddings(self.train_loader)
        train_loader = self._create_loader(train_embeddings, train_concepts, train_labels, batch_size)
        val_loader = self._create_loader(val_embeddings, val_concepts, val_labels, batch_size)
        test_loader = self._create_loader(test_embeddings, test_concepts, test_labels, batch_size)

        return train_loader, val_loader, test_loader
