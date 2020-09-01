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
import imageio


def load_model(LOAD_PATH,model,optimizer):
        state=torch.load(LOAD_PATH)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        return model,optimizer

def tensor_to_im(tensor):
        cv2_formatted=np.transpose(tensor,(1,2,0))
        max0=np.max(cv2_formatted[0])
        max1=np.max(cv2_formatted[1])
        max2=np.max(cv2_formatted[2])
        scaled=np.copy(cv2_formatted)
        scaled[0]=cv2_formatted[0]/max0*179
        scaled[1]=cv2_formatted[1]/max1*255
        scaled[2]=cv2_formatted[2]/max2*255
        scaled=scaled.astype(np.uint8)
        rgbimg = cv2.cvtColor(scaled, cv2.COLOR_HSV2RGB)
        return rgbimg
       
def denormalize(image,channel_mean,channel_std):
    for channel in range(3):
                image[channel]*=std_running_count[channel]
                image[channel]+=mean_running_count[channel]
    
def visualize_recon(BVAE,DAE_net,data,channel_mean,channel_std,SAVE_PATH):
        image_count=data.shape[0]
        original_save_name="original.png"
        reconstructed_save_name="reconstructed.png"
        batch=data.type(torch.cuda.FloatTensor).cuda()
        a=batch.cpu().data.numpy()
        reconstruction,mean,std=BVAE(batch)
        recon=DAE_net(reconstruction)
        new=recon.cpu().data.numpy()
        for image in image_count:
            denormalize(a[image],channel_mean,channel_std)
            denormalize(new[image],channel_mean,channel_std)
        for image in image_count:
            path=os.path.join(SAVE_PATH,str(image))
            os.mkdir(path)
            original_image=tensor_to_im(a[image])
            reconstructed_image=tensor_to_im(new[image])
            original_path=os.path.join(path,original_save_name)
            reconstructed_path=os.path.join(path,reconstructed_save_name)
            cv2.imwrite(original_path,original_image)
            cv2.imwrite(reconstructed_path,reconstructed_image)
            
def latent_traversal(BVAE,DAE_net,data,channel_mean,channel_std,SAVE_PATH):
        images=[]
        pertubs=np.linspace(-3.0,3.0,20)
        batch=data.type(torch.cuda.FloatTensor).cuda()
        reconstruction,mean,std=BVAE1(batch)
        newz=reparametrize(mean,std)
        #newz[0][0]+=pertubs[0]
        #try1=torch.rand(100,32)
        #try1=try1.type(torch.cuda.FloatTensor).cuda()   
        #print(newz.shape)
        for i in range(32):
            for j in range(20):
                y = torch.empty_like(newz).copy_(newz)
                y[0][i]+=pertubs[j]
                recon=BVAE1.decoder(y)
                new=recon.cpu().data.numpy()
                for channel in range(3):
                    new[0][channel]*=std_running_count[channel]
                    new[0][channel]+=mean_running_count[channel]
                changed=np.transpose(new[0],(1,2,0))
                a=changed.astype(np.uint8)
                images.append(a)
            
        path=SAVE_PATH
        os.mkdir(path)
        for i in range(32):
            index=os.path.join(path,str(i))
            os.mkdir(index)
            for j in range(20):
                tosave=images[i*20+j]
                imagename=str(j)+".jpg"
                save_path=os.path.join(index,imagename)
                cv2.imwrite(save_path,tosave)
def make_gif(LOAD_PATH,SAVE_PATH):
        objects=[]
        for image_1 in os.listdir(LOAD_PATH):
            objects.append(imageio.imread(os.path.join(LOAD_PATH,image_1)))
        imageio.mimsave(os.path.join(SAVE_PATH,'movie.gif'), objects)