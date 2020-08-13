import torch.nn as nn
import torch
from dataloader import AliDataset
import cv2
import numpy as np
from torchvision.models import resnet101
import os
import torch.utils.data 
import torchnet.meter as meter
from torch.autograd import Variable
import visdom
os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():
     net = resnet101(pretrained=False)
     pthfile = './resnet101.pth'
     net.load_state_dict(torch.load(pthfile))
     fc_feat_num = net.fc.in_features
     net.fc = nn.Linear(fc_feat_num, 3)

     data_dir = './dataset/train_dataset'
     train_data=AliDataset(data_dir,train=True)
     val_data=AliDataset(data_dir,train=False)
                    #  train_loader = ali_loader(dataset_dir=data_dir, batch_size=4,num_workers=8,use_gpu=True)
     train_dataloader=torch.utils.data.DataLoader(train_data,batch_size=4,num_workers=8)
     val_dataloader=torch.utils.data.DataLoader(val_data,batch_size=4,num_workers=8)
     
     loss_function = nn.CrossEntropyLoss()
     optimizer = torch.optim.SGD(
          net.parameters(),
          lr=0.01,
          momentum=0.9,
          weight_decay=1e-5)
     loss_meter = meter.AverageValueMeter()
     confusion_matrix=meter.ConfusionMeter(2)
     previous_loss=1e100

     for epoch in range(max_epoch=10):
          loss_meter.reset()
          confusion_matrix.reset()
          for step,(img,label)in enumerate(train_dataloader):
               input=Variable(img)
               target=Variable(label)
               input=input.cuda()
               target=target.cuda()
               optimizer.zero_grad()
               score=net(input)
               loss=loss_function(score,target)
               loss.backward()
               optimizer.step()

               loss_meter.add(loss.data[0])
               confusion_matrix.add(score.data,target.data)

               if step%100 == 0:
                    print('train loss: {:.4f}'.format(loss))
          
          net.save()
          val_cm,val_accuracy = val(net,val_dataloader)
           print('val_accuracy: {:.4f}'.format(val_accuracy))
           


if __name__ == "__main__":
    main()