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
            row = row.strip().split(' ')  
            self.labels.append([row[0], int(row[1])])
        #imgs_num = len(self.labels)

    def __getitem__(self, index):
        img_dir = self.labels[index][0]
        data = cv2.imread(img_dir)
        data = cv2.resize(data,(256,144))
        data  = np.transpose(data,(2,0,1)).astype(np.float32)
        
        return data, self.labels[index][0]
 
    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return self.__class__.__name__

def Ali_loader(dataset_dir, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)

    train_list_dir = '/home/apollo/ali-tianchi/train_list.txt'
    val_list_dir = '/home/apollo/ali-tianchi/val_list.txt'
    train_dataset = AliDataset(dataset_dir,train_list_dir) 
    val_dataset = AliDataset(dataset_dir,val_list_dir)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset,val_dataset])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=False,
    )

    return train_loader, val_loader

def Alitest_loader(dataset_dir, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)

    test_list_dir = '/home/apollo/ali-tianchi/test_list1.txt'
    test_dataset = AliDataset(dataset_dir,test_list_dir) 

    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=False,
    )

    return test_loader