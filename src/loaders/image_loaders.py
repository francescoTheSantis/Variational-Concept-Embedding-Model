import torch
from torchvision.datasets import CelebA
from typing import List
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch import nn
from typing import List
import numpy as np
from PIL import Image

import os, re
import requests
import tarfile
from transformers import ViTModel
from torchvision.models import resnet34
from tqdm import tqdm

class EmbeddingExtractor:
    def __init__(self, train_loader, val_loader, test_loader, device='cuda', celeba=False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.celeba = celeba

        # Load ViT model pre-trained on ImageNet
        #self.model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        # Load ResNet34 model pre-trained on ImageNet
        self.model = resnet34(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _extract_embeddings(self, loader):
        """Helper function to extract embeddings for a given DataLoader."""
        embeddings = []
        concepts_list = []
        labels = []

        with torch.no_grad():
            if not self.celeba:
                for images, concepts, targets in tqdm(loader):
                    images = images.to(self.device)
                    # Extract embeddings
                    outputs = self.model(images)
                    # Get the [CLS] token representation
                    #outputs = outputs.last_hidden_state[:, 0, :]
                    outputs = outputs.flatten(start_dim=1)
                    embeddings.append(outputs.cpu())
                    concepts_list.append(concepts.cpu())
                    labels.append(targets.cpu())
            else:
                for images, (concepts, targets) in tqdm(loader):
                    images = images.to(self.device)
                    # Extract embeddings
                    outputs = self.model(images)
                    # Get the [CLS] token representation
                    #outputs = outputs.last_hidden_state[:, 0, :]
                    outputs = outputs.flatten(start_dim=1)
                    embeddings.append(outputs.cpu())
                    concepts_list.append(concepts.cpu())
                    labels.append(targets.cpu())
                
        # Concatenate all embeddings and labels
        embeddings = torch.cat(embeddings, dim=0)
        concepts = torch.cat(concepts_list, dim=0)
        labels = torch.cat(labels, dim=0)

        if len(labels.shape)>1:
            labels = labels.squeeze()

        return embeddings, concepts.float(), labels

    def _create_loader(self, embeddings, concepts, labels, batch_size):
        """Helper function to create a DataLoader from embeddings and labels."""
        dataset = TensorDataset(embeddings, concepts, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    

class CustomDataset(Dataset):
    def __init__(self, mean, std, images, labels, digits):
        
        tensor = [] 
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
    

def MNIST_addition_loader(batch_size, val_size=0.1, seed=42, num_workers=3, root=None, incomplete=False):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    pin_memory=True
    generator = torch.Generator().manual_seed(seed)
    samples_per_permutation = 500

    root_path = os.path.join(root, 'data')
    train_dataset = datasets.MNIST(root=root_path, train=True, download=True)
    test_dataset = datasets.MNIST(root=root_path, train=False, download=True)

    print('Generating pairs for training and test sets...')
    # Create composed training-set
    if not incomplete:
        # Create composed training-set
        unique_pairs = [str(x)+str(y) for x in range(10) for y in range(10)]
        X_train = []
        y_train = []
        c_train = []
        y_train_lab = np.array([x[1] for x in train_dataset])
        y_test_lab = np.array([x[1] for x in test_dataset])
        y_digits = np.array([x[1] for x in test_dataset])
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
    else:
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
        y_digits = np.array([x[1] for x in test_dataset])
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Extracting embeddings...')
    E_extr = EmbeddingExtractor(train_loader, val_loader, test_loader, device=device)
    train_loader, val_loader, test_loader = E_extr.produce_loaders()

    return train_loader, val_loader, test_loader


class CUBDataset(Dataset):
    def __init__(self, root_dir, train=False, selected_concepts=None):
        if selected_concepts==None:
            SELECTED_CONCEPTS = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 
                                45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90,
                                91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 
                                134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 
                                181, 183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 
                                213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249,
                                253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298,
                                299, 304, 305, 308, 309, 310, 311]
        else:
            SELECTED_CONCEPTS = selected_concepts

        self.root_dir = root_dir
        self.train = train

        self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor()
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
    

def CUB200_loader(batch_size, val_size=0.1, seed = 42, root=None, num_workers=3, pin_memory=True, augment=True, shuffle=True, selected_concepts=None):
    generator = torch.Generator().manual_seed(seed) 

    path = os.path.join(root, 'data')
    train_dataset = CUBDataset(root_dir=path, train=True, selected_concepts=selected_concepts)
    test_dataset = CUBDataset(root_dir=path, train=False, selected_concepts=selected_concepts)

    val_size = int(len(train_dataset) * val_size)
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = create_loader(train_loader, batch_size)
    val_loader = create_loader(val_loader, batch_size)
    test_loader = create_loader(test_loader, batch_size)
    
    #if not finetune_backbone:
    #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #    print('Extracting embeddings...')
    #    E_extr = EmbeddingExtractor(train_loader, val_loader, test_loader, device=device)
    #    train_loader, val_loader, test_loader = E_extr.produce_loaders()

    return train_loader, val_loader, test_loader

def create_loader(loader, batch_size):
    img_tensor, concepts_list, labels = [], [], []
    for images, concepts, targets in tqdm(loader):
                        img_tensor.append(images)
                        concepts_list.append(concepts)
                        labels.append(targets)
    img_tensor = torch.cat(img_tensor, dim=0)
    concepts = torch.cat(concepts_list, dim=0).float()
    labels = torch.cat(labels, dim=0)    
    dataset = TensorDataset(img_tensor, concepts, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


#################################### Probably to be eliminated ####################################

# CelebA loader
# Problem of the loader: you need to download the dataset yourself and create a structure like this:
# dataset/
# └── celeba/
#     ├── img_align_celeba/  # This folder should contain the images
#     ├── list_attr_celeba.txt
#     ├── identity_CelebA.txt
#     ├── list_bbox_celeba.txt
#     ├── list_landmarks_align_celeba.txt
#     ├── list_eval_partition.txt
# Additionally differently fom the others, this loader can be use like this:
# for image, (concepts, label) in loaded_train:

#ID:           Attribute:  Balance:
#24             No_Beard  0.834940
#39                Young  0.773617
#2            Attractive  0.512505
#21  Mouth_Slightly_Open  0.483428
#31              Smiling  0.482080
#36     Wearing_Lipstick  0.472436
#19      High_Cheekbones  0.455032
#20                 Male  0.416754
#18         Heavy_Makeup  0.386922
#33            Wavy_Hair  0.319567
#25            Oval_Face  0.284143
#27          Pointy_Nose  0.277445
#1       Arched_Eyebrows  0.266981
#6              Big_Lips  0.240796
#8            Black_Hair  0.239251
#7              Big_Nose  0.234532
#32        Straight_Hair  0.208402
#11           Brown_Hair  0.205194
#3       Bags_Under_Eyes  0.204572
#34     Wearing_Earrings  0.188925
#5                 Bangs  0.151575
#9            Blond_Hair  0.147992
#12       Bushy_Eyebrows  0.142168
#37     Wearing_Necklace  0.122967
#23          Narrow_Eyes  0.115149
#0      5_o_Clock_Shadow  0.111136
#28    Receding_Hairline  0.079778
#38      Wearing_Necktie  0.072715
#29          Rosy_Cheeks  0.065721
#15           Eyeglasses  0.065119
#16               Goatee  0.062764
#13               Chubby  0.057567
#30            Sideburns  0.056511
#10               Blurry  0.050899
#35          Wearing_Hat  0.048460
#14          Double_Chin  0.046688
#26            Pale_Skin  0.042947
#17            Gray_Hair  0.041950
#22             Mustache  0.041545
#4                  Bald  0.022443

#Suggested:
#concept_names = ['Mouth_Slightly_Open', 'Smiling', 'Wearing_Lipstick', 'High_Cheekbones', 'Heavy_Makeup', 'Wavy_Hair', 'Oval_Face', 'Pointy_Nose', 'Arched_Eyebrows', 'Big_Lips']
#class_attributes = ['Male']

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
        concept_names: List[str] = None,
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
            # Default to 'Male' if no class_attributes provided
            self.class_idx = [self.attr_names.index('Male')]
        else:
            # Use the provided class attributes
            self.class_idx = [
                self.attr_names.index(attr) for attr in class_attributes
            ]

        self.attr_names = [string for string in self.attr_names if string]

        if concept_names is not None:
            # Get indices for the concept names
            self.concept_idx = [self.attr_names.index(concept) for concept in concept_names]
    
            # Check for overlap between concept indices and class indices
            overlapping_indices = set(self.concept_idx) & set(self.class_idx)
            if overlapping_indices:
                overlapping_names = [self.attr_names[i] for i in overlapping_indices]
                raise ValueError(f"Overlap detected between concept names and class attributes: {overlapping_names}")
        else:
            # Determine concept attribute names based on class attributes
            self.concept_idx = [i for i in range(len(self.attr_names)) if i not in self.class_idx]

        self.concept_attr_names = [self.attr_names[i] for i in self.concept_idx]
        self.task_attr_names = [self.attr_names[i] for i in self.class_idx]

    def __getitem__(self, index: int):
        image, attributes = super(CelebADataset, self).__getitem__(index)

        # Extract the target (y) based on the class index
        y = torch.stack([attributes[i] for i in self.class_idx])

        # Extract concept attributes, excluding the class attributes
        concept_attributes = torch.stack([attributes[i] for i in self.concept_idx])

        return image, concept_attributes, y


def CelebA_loader(batch_size, val_size=0.1, seed = 42, root=None, class_attributes=['Male'], concept_names=['Straight_Hair'], num_workers=3, pin_memory=True, shuffle=True):
    generator = torch.Generator().manual_seed(seed) 

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
        ])

    path = os.path.join(root, 'data')

    #Download in the following lines is set to false due to a known torchvision issue which has 
    #never been solved. YOU HAVE TO DOWNLOAD Celeba MANUALLY and unzip it in the folder you use as root
    #https://stackoverflow.com/questions/70896841/error-downloading-celeba-dataset-using-torchvision
    train_dataset = CelebADataset(root=path, split="train", class_attributes=class_attributes, concept_names=concept_names, transform=train_transform, download=False)
    test_dataset = CelebADataset(root=path, split="test", class_attributes=class_attributes, concept_names=concept_names, transform=test_transform, download=False)

    val_size = int(len(train_dataset) * val_size)
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Extracting embeddings...')
    E_extr = EmbeddingExtractor(train_loader, val_loader, test_loader, device=device)
    train_loader, val_loader, test_loader = E_extr.produce_loaders()

    return train_loader, val_loader, test_loader

