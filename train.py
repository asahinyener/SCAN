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
from model.py import DAE,BVAE


class DAEdata(data.Dataset):
        def __init__(self,hsv_images):
            self.hsv_images=hsv_images
            self.image_size=hsv_images[0].shape
        def __len__(self):
            return len(self.hsv_images)
        def __getitem__(self,index):
            ground_truth=self.hsv_images[index]        
            h0,h1=sorted(random.sample(range(0,80),2))
            w0,w1=sorted(random.sample(range(0,80),2))
            blocked_image=np.copy(ground_truth)
            blocked_image[:,h0:h1,w0:w1]=0.0
            return ground_truth,blocked_image

class VAEdata(data.Dataset):
        def __init__(self,hsv_images):
            self.hsv_images=hsv_images
            self.image_size=hsv_images[0].shape
        def __len__(self):
            return len(self.hsv_images)
        def __getitem__(self,index):
            ground_truth=self.hsv_images[index]        
            return ground_truth
def split_train_test(data,train_size):
    return data[:train_size],data[train_size:]
def train_dae(DAEnet,optim_dae,train_data_generator,test_data_generator,criterion,check_point_dir,epoch_num,writer,output_file_path):
        for epoch in range(epoch_num):
            running_loss=0.0
            for i,batch in enumerate(data_generator,0):
                a,b=batch
                a=a.type(torch.cuda.FloatTensor).cuda()
                b=b.type(torch.cuda.FloatTensor).cuda()
                decoded=DAEnet(b)
                loss=criterion(decoded,a)
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
            if(epoch%50==49):
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
            running_loss=0.0
            for i,batch in enumerate(train_data_generator,0):
                batch=batch.type(torch.cuda.FloatTensor).cuda()
                reconstruction,mean,std=BVAE(batch)
                loss=BVAE.compute_loss(batch,reconstruction,mean,std)
                writer.add_scalar("Loss/train", loss, epoch)
                running_loss+=loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                if(i%20==19):
                    with open(output_file_path,"a") as ofile:
                        ofile.write('[%d, %5d] loss: %.3f \n' %
                          (epoch + 1, i + 1, running_loss /20))
                        ofile.close()
                    running_loss = 0.0
            if(epoch%50==49):
                print("saving!")
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