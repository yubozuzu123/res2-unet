
########train.py
from datetime import datetime
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os 
from model.res2_unet import res2net50_v1b_26w_4s 
import cv2
import scipy.ndimage as ndimage
 
from model.data_loader_whu import train_dataloader,eval_dataloader


def RobertsAlogrithm(image):
    edges = image - ndimage.morphology.binary_erosion(image)
    return edges
def focal_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred+1e-10) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred+1e-10) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss 
def boundary_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''

    pred_num=pred.cpu()
    pred_edge=RobertsAlogrithm(pred_num.detach().numpy())
    gt_edge=RobertsAlogrithm(gt.cpu().detach().numpy())
    
    sum1=(pred_edge-gt_edge)*(pred_edge-gt_edge)
    sum2=pred_edge*pred_edge
    sum3=gt_edge*gt_edge
   
    return sum1.sum()/(sum2.sum()+sum3.sum()+1e-10)
    
def adjust_learning_rate(optimizer, epoch,lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     
                      
def train(epo_num=100):
 
    #vis = visdom.Visdom()
    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model=res2net50_v1b_26w_4s(pretrained=True)
    model = model.to(device)
    criterion = nn.BCELoss().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.9,
                                weight_decay=0.0001)
 
    all_train_iter_loss = []
    all_test_iter_loss = []
 
    # start timing
    print("start training....")
    prev_time = datetime.now()
    for epo in range(epo_num):
        
        train_loss = 0
        model.train()
        adjust_learning_rate(optimizer, epo,0.01)
        for index, (ls,ls_msk) in enumerate(train_dataloader):
           
            ls = ls.to(device)
            ls_msk = ls_msk.to(device)
            
            optimizer.zero_grad()
            output = model(ls)
            output = torch.sigmoid(output) 
            
            
            boundary_l=boundary_loss(output,ls_msk)
            loss=2*criterion(output, ls_msk)+boundary_l
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()
 
          
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for index, (ls,ls_msk) in enumerate(eval_dataloader):
 
                ls = ls.to(device)
                ls_msk = ls_msk.to(device)
 
                optimizer.zero_grad()
                output = model(ls)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                boundary_l=boundary_loss(output,ls_msk)
                loss = 2*criterion(output, ls_msk)+boundary_l
                
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

        
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
 
        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(eval_dataloader), time_str))
        
        
 
        if np.mod(epo,10) == 0:
            torch.save(model, '/home/boyu/Res2Net/model/res2_unet_whu_data_2bce1b_{}.pt'.format(epo))
            print('saving model/res2_unet_whu_data_2bce1b_{}.pt'.format(epo))
 
 
if __name__ == "__main__":
    train(epo_num=3000)
