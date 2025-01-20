import torch
from torchvision.datasets import CelebA
from typing import List
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch import nn
import numpy as np
from PIL import Image

import os, re
import requests
import tarfile


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
    samples_per_permutation = 500
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


class CUBDataset(Dataset):
    def __init__(self, root_dir, train=False):
        SELECTED_CONCEPTS = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 
                             45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90,
                             91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 
                             134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 
                             181, 183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 
                             213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249,
                             253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298,
                             299, 304, 305, 308, 309, 310, 311]

        self.root_dir = root_dir
        self.train = train

        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.247, 0.243, 0.261)

        '''
        if self.train:
            self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=10), 
                        transforms.Resize((280, 280)),  # image_size + 1/4 * image_size
                        transforms.RandomResizedCrop((224, 224)),
                        transforms.ToTensor()
                    ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor()
                ]) 
        '''
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]) 
        
        if os.path.isdir(self.root_dir + "/CUB_200_2011") is False:
            url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
            file_name = "CUB_200_2011.tgz"
            self._download_file(url, file_name)
            self._extract_file(file_name, root_dir)
            os.remove(file_name)

        # Parse the dataset files
        dataset_dir = root_dir + "/CUB_200_2011"
        self.image_paths_train = []
        self.labels_train = []
        self.image_ids_train = []
        self.image_paths_test = []
        self.labels_test = []
        self.image_ids_test = []

        with open(os.path.join(dataset_dir, "images.txt"), "r") as img_file:
            image_lines = img_file.readlines()
        with open(os.path.join(dataset_dir, "image_class_labels.txt"), "r") as label_file:
            label_lines = label_file.readlines()
        with open(os.path.join(dataset_dir, "train_test_split.txt"), "r") as split_file:
            split_lines = split_file.readlines()

        # Initialize a dictionary to hold the boolean arrays for each image
        self.image_attributes = {}
        with open(os.path.join(dataset_dir, "./attributes/image_attribute_labels.txt"), "r") as file:
            for line in file:
                matches = re.findall(r"\d+\.\d+|\d+", line)
                image_id, attribute_id, is_present = matches[0], matches[1], matches[2] #line.strip().split(" ")
                image_id = int(image_id)
                attribute_id = int(attribute_id)
                is_present = int(is_present)
                if image_id not in self.image_attributes:
                    cnt = 0
                    self.image_attributes[image_id] = np.zeros(len(SELECTED_CONCEPTS), dtype=float)
                if attribute_id in SELECTED_CONCEPTS:
                    self.image_attributes[image_id][cnt] = float(is_present)
                    cnt += 1


        # Extract image paths and labels
        for img_line, label_line, split_line in zip(image_lines, label_lines, split_lines):
            img_id, img_path = img_line.strip().split(" ")
            label_id, label = label_line.strip().split(" ")
            img2_id, split_id = split_line.strip().split(" ")
            assert img_id == label_id == img2_id # Ensure consistent IDs
            if split_id == '1':
                self.image_ids_train.append(int(img_id))
                self.image_paths_train.append(os.path.join(dataset_dir, "images", img_path))
                self.labels_train.append(int(label) - 1)  # Convert to zero-based index
            else:
                self.image_ids_test.append(int(img_id))
                self.image_paths_test.append(os.path.join(dataset_dir, "images", img_path))
                self.labels_test.append(int(label) - 1)  # Convert to zero-based index


    def __len__(self):
        if self.train:
            return len(self.image_paths_train)
        return len(self.image_paths_test)
    
    # Step 1: Download the file
    def _download_file(self, url, file_name):
        print(f"Downloading {file_name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(file_name, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {file_name} successfully.")

    # Step 2: Extract the tar.gz file
    def _extract_file(self, file_name, output_dir):
        print(f"Extracting {file_name}...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"Extracted files to {output_dir}.")

    def __getitem__(self, idx):
        if self.train:
            img_path = self.image_paths_train[idx]
            label = self.labels_train[idx]
            concepts = torch.from_numpy(self.image_attributes[self.image_ids_train[idx]])
        else:
            img_path = self.image_paths_test[idx]
            label = self.labels_test[idx]
            concepts = torch.from_numpy(self.image_attributes[self.image_ids_test[idx]])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, concepts, label
    

def CUB200_loader(batch_size, val_size=0.1, seed = 42, dataset='./datasets/', num_workers=3, pin_memory=True, augment=True, shuffle=True):
    generator = torch.Generator().manual_seed(seed) 

    train_dataset = CUBDataset(root_dir='./dataset', train=True)
    test_dataset = CUBDataset(root_dir='./dataset', train=False)

    val_size = int(len(train_dataset) * val_size)
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
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

        train_loader = self._create_loader(train_embeddings, train_concepts, train_labels, batch_size)
        val_loader = self._create_loader(val_embeddings, val_concepts, val_labels, batch_size)
        test_loader = self._create_loader(test_embeddings, test_concepts, test_labels, batch_size)

        return train_loader, val_loader, test_loader
