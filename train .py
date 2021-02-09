import cv2
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from model import DAE,BVAE
import utils

def train_dae(DAEnet,optim_dae,train_data_generator,test_data_generator,criterion,check_point_dir,epoch_num,writer,output_file_path):
        for epoch in range(epoch_num):
            running_loss=0.0
            for i,batch in enumerate(train_data_generator,0):
                a,b=batch
                a=a.type(torch.cuda.FloatTensor).cuda()
                b=b.type(torch.cuda.FloatTensor).cuda()
                decoded=DAEnet(b)
                loss=criterion(decoded,a)
                #loss=-(a * torch.log(decoded) + (1 - a) * torch.log(1 - decoded)).sum() / 32
                writer.add_scalar("Loss/dae_train", loss, epoch)
                optim_dae.zero_grad()
                loss.backward()
                optim_dae.step()
                if(i%20==19):
                    with open(output_file_path,"a") as ofile:
                        ofile.write('[%d, %5d] loss: %.3f \n' %
                          (epoch + 1, i + 1, running_loss /20))
                        ofile.close()
                    running_loss = 0.0
            if(epoch%100==99):
                #print("saving!")
                state = {
                    'checkpoint_num': epoch,
                    'state_dict': DAEnet.state_dict(),
                    'optimizer': optim_dae.state_dict(),                
                }
                path=str(epoch+1)+".pt"
                saves=os.path.join(check_point_dir,path)
                torch.save(state,saves)
                for i,batch in enumerate(test_data_generator,0):
                    a,b=batch
                    a=a.type(torch.cuda.FloatTensor).cuda()
                    b=b.type(torch.cuda.FloatTensor).cuda()
                    decoded=DAEnet(b)
                    loss=criterion(decoded,a)
                    writer.add_scalar("Loss/dae_test", loss, epoch)
        writer.flush()

def train_bvae(BVAE,optim,train_data_generator,test_data_generator,check_point_dir,epoch_num,writer,output_file_path):
        for epoch in range(epoch_num):
            #running_loss=0.0
            for i,batch in enumerate(train_data_generator,0):
                batch=batch.type(torch.cuda.FloatTensor).cuda()
                reconstruction,mean,std=BVAE(batch)
                loss=BVAE.compute_loss(batch,reconstruction,mean,std)
                writer.add_scalar("Loss/train", loss, epoch)
                #running_loss+=loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                """if(i%20==19):
                    with open(output_file_path,"a") as ofile:
                        ofile.write('[%d, %5d] loss: %.3f \n' %
                          (epoch + 1, i + 1, running_loss /20))
                        ofile.close()
                    running_loss = 0.0"""
            if(epoch%100==99):
                #print("saving!")
                state = {
                    'checkpoint_num': epoch,
                    'state_dict': BVAE.state_dict(),
                    'optimizer': optim.state_dict(),                
                }
                path=str(epoch+1)+".pt"
                saves=os.path.join(check_point_dir,path)
                torch.save(state,saves)
                for i,batch in enumerate(test_data_generator,0):
                    batch=batch.type(torch.cuda.FloatTensor).cuda()
                    reconstruction,mean,std=BVAE(batch)
                    loss=BVAE.compute_loss(batch,reconstruction,mean,std)
                    writer.add_scalar("Loss/test", loss, epoch)
        writer.flush()
def train_scan(SCAN_net,optim_scan,train_data_generator,test_data_generator,check_point_dir,epoch_num,writer,output_file_path):
    losses=[]
    for epoch in range(epoch_num):
        for batchid,batch in enumerate(train_data_generator,0):
            one_hot,image=batch
            one_hot=one_hot.type(torch.cuda.FloatTensor).cuda()
            image=image.type(torch.cuda.FloatTensor).cuda()
            recon,m,s=SCAN_net(one_hot)
            loss,kld=SCAN_net.compute_loss(image,one_hot,recon,m,s)
            losses.append(kld)
            writer.add_scalar("Loss/train/scan",loss,epoch)
            optim_scan.zero_grad()
            loss.backward()
            #optim_scan_zero_grad()
            optim_scan.step()
        if(epoch%50==49):
                    #print("saving!")
                    state = {
                        'checkpoint_num': epoch,
                        'state_dict': SCAN_net.state_dict(),
                        'optimizer': optim_scan.state_dict(),                
                    }
                    path_now=str(epoch+1)+".pt"
                    saves=os.path.join(check_point_dir,path_now)
                    torch.save(state,saves)
                    for i,batch in enumerate(test_data_generator,0):
                        one_hot,image=batch
                        one_hot=one_hot.type(torch.cuda.FloatTensor).cuda()
                        image=image.type(torch.cuda.FloatTensor).cuda()
                        recon,m,s=SCAN_net(one_hot)
                        loss,kld=SCAN_net.compute_loss(image,one_hot,recon,m,s)
                        writer.add_scalar("Loss/test/scan", loss, epoch)
    writer.flush()
    return losses
def train_recomb(SCAN_Recomb,optim_recomb,train_data_generator,check_point_dir):
    for batch_id,batch in enumerate(train_data_generator,0):
        op_type,y0,y1,y_ground_truth,images=batch
        #print(op_type)
        #print(one_hot_to_label(y0[0]))
        #print(one_hot_to_label(y1[0]))
        #print(one_hot_to_label(y_ground_truth[0]))

        #img.append(images)
        
        op_type=op_type.type(torch.cuda.FloatTensor)
        y0=y0.type(torch.cuda.FloatTensor).cuda()
        y1=y1.type(torch.cuda.FloatTensor).cuda()
        y_ground_truth=y_ground_truth.type(torch.cuda.FloatTensor).cuda()
        images=images.type(torch.FloatTensor).cuda()
        out0,out1=SCAN_Recomb(y0,y1,op_type.unsqueeze(2))
        loss=recomb.compute_loss(images,out0,out1,y_ground_truth)
        optim_recomb.zero_grad()
        #losses.append(loss.item())
        loss.backward()
        optim_recomb.step()
    utils.save_model(SCAN_Recomb,optim_recomb,check_point_dir,"SCAN_Recomb.pt")