import cv2
import os
#import numpy as np
#import random
#from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x
        
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)
def down_conv_block(in_layer,out_layer,padding=True,kernel_size=4,activation=True):
        if(activation):
            return nn.Sequential(
            nn.Conv2d(in_channels=in_layer,out_channels=out_layer,kernel_size=kernel_size,stride=2,padding=1),
            #PrintLayer(), #,padding=1
            nn.ELU()    
        )
        else:
            return nn.Sequential(
            nn.Conv2d(in_channels=in_layer,out_channels=out_layer,kernel_size=kernel_size,stride=2,padding=1),
            #PrintLayer(), #,padding=1
            #nn.ELU()    
        )
            
            
def up_conv_block(in_layer,out_layer,kernel_size=4,activation=True):
    if(activation):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_layer,out_channels=out_layer,kernel_size=kernel_size,stride=2,padding=1),
                #PrintLayer(),
                nn.ELU()
            )
    else:
            return nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_layer,out_channels=out_layer,kernel_size=kernel_size,stride=2,padding=1),
                    #PrintLayer(),
                    #nn.ELU()
                )
class DAE(nn.Module):
    def __init__(self,in_size,encoder_sizes,decoder_sizes,encoder_activation,decoder_activation,bottlenecksize,batch_size):
        super().__init__()
        self.encoder_sizes=[in_size, *encoder_sizes]
        self.decoder_sizes=[*decoder_sizes,in_size]
        self.encoder=nn.Sequential(
                                        *[down_conv_block(in_layer,out_layer,activation=activation) for in_layer,out_layer,activation in zip(self.encoder_sizes,self.encoder_sizes[1:],encoder_activation)],
                                        nn.Flatten(),
                                        nn.Linear(5*5*self.encoder_sizes[-1],bottlenecksize)
                                        )
        self.decoder=nn.Sequential(
                    nn.Linear(bottlenecksize,5*5*self.decoder_sizes[0]),
                    View((-1,64,5,5)),                   
                    *[up_conv_block(in_layer,out_layer,activation=activation) for in_layer,out_layer,activation in zip(self.decoder_sizes,self.decoder_sizes[1:],decoder_activation)],
                    )
    def forward(self,blocked_image):
        encoded=self.encoder(blocked_image)
        decoded=self.decoder(encoded)
        return decoded
def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps
class BVAE(nn.Module):
    def __init__(self,in_size,encoder_sizes,decoder_sizes,encoder_activation,decoder_activation,latent_size,beta,dae_net):
        super().__init__()
        self.encoder_sizes=[in_size, *encoder_sizes]
        self.decoder_sizes=[*decoder_sizes,in_size]
        self.latent_size=latent_size
        self.beta=beta
        self.dae_net=dae_net
        self.encoder=nn.Sequential(
                                        *[down_conv_block(in_layer,out_layer,activation=activation) for in_layer,out_layer,activation in zip(self.encoder_sizes,self.encoder_sizes[1:],encoder_activation)],
                                        nn.Flatten(),
                                        nn.Linear(5*5*self.encoder_sizes[-1],256),
                                        nn.ReLU(),
                                        nn.Linear(256,latent_size*2)
                                        )
        self.decoder=nn.Sequential(
                    nn.Linear(latent_size,256),
                    nn.ReLU(),
                    nn.Linear(256,5*5*self.encoder_sizes[-1]),
                    nn.ReLU(),                    
                    View((-1,64,5,5)),                   
                    *[up_conv_block(in_layer,out_layer,activation=activation) for in_layer,out_layer,activation in zip(self.decoder_sizes,self.decoder_sizes[1:],decoder_activation)],
                    )
        #for in_layer,out_layer,activation in zip(self.decoder_sizes,self.decoder_sizes[1:],decoder_activation):
        #    print(in_layer,out_layer,activation)
    def forward(self,x):
        parametrized=self.encoder(x)
        mean=parametrized[:,:self.latent_size]
        #print(mean.shape)
        std=parametrized[:,self.latent_size:]
        z=reparametrize(mean,std)
        reconstruction=self.decoder(z)
        return reconstruction,mean,std
    def compute_loss(self,original,reconstruction,mu,logvar):
        batch_size = original.size(0)
        reconstruction_loss= F.mse_loss(self.dae_net(reconstruction), self.dae_net(original), size_average=False).div(batch_size)
        #batch_size = mu.size(0)
        #assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        klds=klds.mean(0).sum()
        return self.beta * klds  +reconstruction_loss
