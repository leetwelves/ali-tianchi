#coding:utf8
from config import opt
import os
import torch as t
from torch.autograd import Variable
import models
from data.dataloader import Ali_loader,Alitest_loader
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
from torchvision.models import resnet101
from collections import OrderedDict
os.environ['CUDA_VISIBLE_DEVICES']='0'



def  main():
    
    train()

@t.no_grad() # pytorch>=0.5
def test(epoch,**kwargs):
    opt._parse(kwargs)

    # configure model
    model=resnet101(pretrained=False)
    fc_feat_num = model.fc.in_features
    model.fc = t.nn.Linear(fc_feat_num, 3)
    pthfile='./checkpoints/model_state_{}.pth'.format(epoch)
    model.load_state_dict(t.load(pthfile))
    model.cuda()
   
    # data
    data_dir='/home/apollo/ali-tianchi/dataset/test_dataset'
    test_loader=Alitest_loader(dataset_dir=data_dir,batch_size=opt.batch_size,num_workers=opt.num_workers,use_gpu=opt.use_gpu)

    results=[]
    for ii ,(data,path) in enumerate(test_loader):
<<<<<<< HEAD
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score,dim=1)[:,0].detach().tolist()
        
        probability = t.nn.functional.softmax\
            (score,dim=1).data.cpu().numpy()
=======
        input=t.autograd.Variable(data,volatile =True)
        input=input.cuda()
        score=model(input)
        #print(score)
        probability = t.nn.functional.softmax\
            (score).data.cpu().numpy()
>>>>>>> c429726b3810641e1d56e891720cdbbc16848a66
        probability = probability.argmax(1)
        batch_results = [(path_,probability_)
            for path_,probability_ in zip(path,probability)]
        results += batch_results
    write_csv(results,'./results/model_test_{}.csv'.format(epoch))
<<<<<<< HEAD
 
=======
>>>>>>> c429726b3810641e1d56e891720cdbbc16848a66
    



def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
       # writer.writerow(['id','label'])
        writer.writerows(results)
    
def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)

    # step1: configure model
    model = resnet101(pretrained=False)
    pthfile = './resnet101.pth'
    model.load_state_dict(t.load(pthfile))
    # for param in model.parameters():
    #     param.requires_grad = True
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    fc_feat_num = model.fc.in_features
    model.fc = t.nn.Linear(fc_feat_num, 3)
    # model.train()
    model.cuda()

   

    # step2: data
    data_dir='/home/apollo/ali-tianchi/dataset/train_dataset'
    train_loader,val_loader=Ali_loader(dataset_dir=data_dir,batch_size=opt.batch_size,num_workers=opt.num_workers,use_gpu=opt.use_gpu)
    
    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer=t.optim.SGD( model.parameters(),
                                                        lr=lr,
                                                        momentum=0.9,
                                                        weight_decay=opt.weight_decay)
<<<<<<< HEAD
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma = 0.1, last_epoch=-1)
    
=======

>>>>>>> c429726b3810641e1d56e891720cdbbc16848a66
   
    
        
    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(3)
    # previous_loss = 1e10

    # train
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        confusion_matrix.reset()
       

        for ii,(data,label) in tqdm(enumerate(train_loader)):

                # train model 
                #input = data.to(opt.device)
                #target = label.to(opt.device)
                #input=Variable(data)
                #target=Variable(label)
                input=data
                target=label

                input=input.cuda()            
                target=target.cuda()
                
                optimizer.zero_grad()
                score = model(input)
                loss = criterion(score,target)
                loss.backward()
                optimizer.step()
                # meters update and visualize
                loss_meter.add(loss.item())
                #detach 一下更安全保险
                confusion_matrix.add(score.detach(), target.detach()) 

                if (ii + 1)%opt.print_freq == 0:
                        vis.plot('loss', loss_meter.value()[0])
                        print('train loss: {:.4f}'.format(loss))
                     # 进入debug模式
                        # if os.path.exists(opt.debug_file):
                        #   import ipdb
                        #  ipdb.set_trace()

        #state = OrderedDict(
        #        [('state_dict', model.state_dict()),
         #       ('optimizer',optimizer.state_dict()) ,
          #      ('epoch',epoch)])
        t.save(model.state_dict(),'./checkpoints/model_state_{}.pth'.format(epoch))

            # validate and visualize
        val_cm,val_accuracy = val(model,val_loader)

        vis.plot('val_accuracy',val_accuracy)
        print('val accurary: {:.4f}'.format(val_accuracy))
        vis.log("epoch:{epoch},lr:{optlr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch = epoch,optlr=optimizer.state_dict()['param_groups'][0]['lr'],loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        
        # update learning rate
        scheduler.step()
        # if loss_meter.value()[0] > previous_loss:          
        #     lr = lr * opt.lr_decay
        #     # 第二种降低学习率的方法:不会有moment等信息的丢失
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        

<<<<<<< HEAD
        # previous_loss = loss_meter.value()[0]
    
=======
        previous_loss = loss_meter.value()[0]
>>>>>>> c429726b3810641e1d56e891720cdbbc16848a66
       # test(epoch=epoch)

@t.no_grad()
def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval() 
    confusion_matrix = meter.ConfusionMeter(3)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        with t.no_grad():
            score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]+cm_value[2][2]) / (cm_value.sum())
    return confusion_matrix, accuracy

def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
   # import fire
  #  fire.Fire()
<<<<<<< HEAD
    #main()
    test(epoch=38)
=======
    main()
    # test(epoch=38)
>>>>>>> c429726b3810641e1d56e891720cdbbc16848a66
