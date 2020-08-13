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
    def __init__(self, dataset_dir):

        path = os.path.join(dataset_dir, 'amap_traffic_annotations_train.json')
        file = open(path, encoding='utf-8')
        file = json.loads(file.read())
        self.labels = []
        for sample in file['annotations']:
            id = sample['id']
            status = sample['status']
            for frame in sample['frames']:
                frame_name = os.path.join(dataset_dir,id,frame['frame_name'])
                self.labels.append([frame_name, status])
        self.length = len(self.labels)
    def __getitem__(self, index):
        img_dir = self.labels[index][0]
        img = cv2.imread(img_dir)
        img = cv2.resize(img,(224,224))
        img  = np.transpose(img,(2,0,1)).astype(np.float32)
        
        return img, self.labels[index][1]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__

def ali_loader(dataset_dir, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)


    train_dataset = AliDataset(dataset_dir)
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )

    return train_loader
