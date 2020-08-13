import torch.nn as nn
import torch
from dataloader import ali_loader
import cv2
import numpy as np
from torchvision.models import resnet101
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
def main():
     net = resnet101(pretrained=False)
     pthfile = './resnet101.pth'
     net.load_state_dict(torch.load(pthfile))
     fc_feat_num = net.fc.in_features
     net.fc = nn.Linear(fc_feat_num, 3)
     data_dir = './dataset/train_dataset'
     train_loader = ali_loader(dataset_dir=data_dir, batch_size=4,num_workers=8,use_gpu=True)
     loss_function = nn.CrossEntropyLoss()
     optimizer = torch.optim.SGD(
          net.parameters(),
          lr=0.01,
          momentum=0.9,
          weight_decay=1e-5)
     net.train()
     net.cuda()
     for step, (img, label) in enumerate(train_loader):
          img = img.cuda()
          label = label.cuda()
          optimizer.zero_grad()
          outputs = net(img)
          loss = loss_function(outputs,label)
          loss.backward()
          optimizer.step()
          if step % 100 ==0:
               print('train loss: {:.4f}'.format(loss))

if __name__ == "__main__":
    main()