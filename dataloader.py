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
    def __init__(self, dataset_dir,train=True,test=False):
        self.test=test
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
        if self.test:
             self.labels=sorted(self.labels,key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
             self.labels=sorted(self.labels,key=lambda x: int(x.split('.')[-2]))
        
        imgs_num = len(self.labels)

        if self.test:
            self.labels=self.labels
        elif train:
            self.labels=self.labels[:int(0.7*imgs_num)]
        else :
            self.labels=self.labels[int(0.7*imgs_num):]
                
    
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


