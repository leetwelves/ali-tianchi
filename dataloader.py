# coding: utf-8

import os
import numpy as np

import torch
import torch.utils.data
import h5py    
from torchvision import transforms
import cv2
import json


class AliDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, data_list):
        path = os.path.join(dataset_dir, data_list)
        self.labels = []
        file = open(path,'r')
        for row in file:
            row = file.strip().split(' ')
            self.labels.append(row[0], int(row[1]))
        imgs_num = len(self.labels)

    def __getitem__(self, index):
        img_dir = self.labels[index][0]
        img = cv2.imread(img_dir)
        img = cv2.resize(img,(224,224))
        img  = np.transpose(img,(2,0,1)).astype(np.float32)
        
        return img, self.labels[index][1]
 
    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return self.__class__.__name__

def Ali_loader(dataset_dir, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)

    train_list_dir = '/Users/stephan/Desktop/阿里天池/train_list.txt'
    test_list_dir = '/Users/stephan/Desktop/阿里天池/test_list.txt'
    train_dataset = AliDataset(dataset_dir, train_list_dir)
    test_list_dir = AliDataset(dataset_dir, test_list_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=False,
    )

    return train_loader, test_loader