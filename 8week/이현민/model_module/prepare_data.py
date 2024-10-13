

import os
import zipfile

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(train_dir, 
                       test_dir, 
                       train_transform, 
                       test_transform,
                       batch_size):

    with zipfile.ZipFile("/Users/ihyeonmin/Documents/Fool-AIStudy/Fool-AIStudy/8week/이현민/Food101.zip", "r") as zip_f:
        print("Unzipping the dataset.") 
        zip_f.extractall("data_food101/data")

        
    # 1) image directories -> image transformation -> ImageFolder
    
    train_imgfolder = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_imgfolder  = datasets.ImageFolder(root=test_dir,  transform=test_transform)

    class_names = train_imgfolder.classes

    
    # 2) ImageFolder -> DataLoader (iterable for mini-batches)
    
    torch.manual_seed(42)
    
    train_dataloader = DataLoader(train_imgfolder,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True # GPU로의 데이터 로딩 속도 향상 @ https://bit.ly/3B70xIV
    )
    test_dataloader = DataLoader(test_imgfolder,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True # GPU로의 데이터 로딩 속도 향상 @ https://bit.ly/3B70xIV
    )
    
    return train_dataloader, test_dataloader, class_names
