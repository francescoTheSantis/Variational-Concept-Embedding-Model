import torch
from torchvision.datasets import CelebA
from typing import List
from transformers import ViTFeatureExtractor, ViTModel
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
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
            #transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3), 
            #transforms.ToTensor(),
            #transforms.Normalize(self.mean, self.std)
        ])

        # Initialize the ViT feature extractor and model
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        label = self.labels[idx]
        image = self.transform(image)   
        # Extract features using ViT
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0).mean(dim=0)  # Average pooling

        digits = np.zeros(10, dtype=float)
        digits[self.digits[idx]] = 1 
        return embedding, digits, label
    

def MNIST_addition_loader(batch_size, root, val_size=0.1, seed=42, num_workers=3, pin_memory=True, shuffle=True):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    
    # fix the seed for both pytorch generator and numpy.random
    generator = torch.Generator().manual_seed(seed) 
    np.random.seed(seed)

    train_dataset = datasets.MNIST(root=root, train=True, download=True)
    test_dataset = datasets.MNIST(root=root, train=False)

    # Create composed training-set
    unique_pairs = [str(x)+str(y) for x in range(10) for y in range(10)]
    X_train = []
    y_train = []
    c_train = []
    y_train_lab = np.array([x[1] for x in train_dataset])
    y_test_lab = np.array([x[1] for x in test_dataset])
    # y_digits = np.array([x[1] for x in test_dataset])
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
      
    
    # Create the composed dataset (two images concatenated over the x-axis)
    unique_pairs = [str(x)+str(y) for x in range(10) for y in range(10)]
    test_set_pairs = []
    while(len(test_set_pairs) < 10):
        pair_to_add = np.random.choice(unique_pairs)
        if pair_to_add not in test_set_pairs:
            test_set_pairs.append(pair_to_add)
    train_set_pairs = list(set(unique_pairs) - set(test_set_pairs))
    assert(len(test_set_pairs) == 10)
    assert(len(train_set_pairs) == 90)
    for test_set in test_set_pairs:
        assert(test_set not in train_set_pairs)
        print("%s not in training set." % test_set)
    X_train = []
    y_train = []
    c_train = []
    y_train_lab = np.array([x[1] for x in train_dataset])
    y_test_lab = np.array([x[1] for x in test_dataset])
    # y_digits = np.array([x[1] for x in test_dataset])
    samples_per_permutation = 1000
    for train_set_pair in train_set_pairs:
        for _ in range(samples_per_permutation):
            rand_i = np.random.choice(np.where(y_train_lab == int(train_set_pair[0]))[0])
            rand_j = np.random.choice(np.where(y_train_lab == int(train_set_pair[1]))[0])
            temp_image = np.zeros((28,56), dtype="uint8")
            temp_image[:,:28] = train_dataset[rand_i][0]
            temp_image[:,28:] = train_dataset[rand_j][0]
            X_train.append(temp_image)
            y_train.append(y_train_lab[rand_i] + y_train_lab[rand_j])
            c_train.append([y_train_lab[rand_i], y_train_lab[rand_j]])
    X_test = []
    y_test = []
    c_test = []
    for test_set_pair in test_set_pairs:
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader