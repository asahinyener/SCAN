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
def down_conv_block_bvae(in_layer,out_layer,padding=True,kernel_size=4,activation=True):
        if(activation):
            return nn.Sequential(
            nn.Conv2d(in_channels=in_layer,out_channels=out_layer,kernel_size=kernel_size,stride=2,padding=1),
            #PrintLayer(), #,padding=1
            nn.ReLU()    
        )
        else:
            return nn.Sequential(
            nn.Conv2d(in_channels=in_layer,out_channels=out_layer,kernel_size=kernel_size,stride=2,padding=1),
            #PrintLayer(), #,padding=1
            #nn.Sigmoid()    
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
                    #nn.Sigmoid()
                )
def up_conv_block_bvae(in_layer,out_layer,kernel_size=4,activation=True):
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
                    #nn.Sigmoid()
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
    def encode_to_latent(self,x):
        parametrized=self.encoder(x)
        mean=parametrized[:,:self.latent_size]
        std=parametrized[:,self.latent_size:]
        return mean,std
class SCAN(nn.Module):
    def __init__(self,in_size,fc_size,latent_size,beta,lmbd,bvae):
        super().__init__()
        self.bvae=bvae
        self.beta=beta
        self.lmbd=lmbd
        self.latent_size=latent_size
        self.encoder=nn.Sequential(
            nn.Linear(in_size,fc_size),
            nn.ReLU(),
            nn.Linear(fc_size,latent_size*2),
        )
        self.decoder=nn.Sequential(
            nn.Linear(latent_size,fc_size),
            nn.ReLU(),
            nn.Linear(fc_size,in_size),
            nn.Sigmoid()
        )
    def forward(self,one_hot):
        distributions=self.encoder(one_hot)
        mean=distributions[:,:self.latent_size]
        std=distributions[:,self.latent_size:]
        z=reparametrize(mean,std)
        reconstruction=self.decoder(z)
        return reconstruction,mean,std
    def image_from_symbol(self,one_hot):
        distributions=self.encoder(one_hot)
        mean=distributions[:,:self.latent_size]
        std=distributions[:,self.latent_size:]
        z=reparametrize(mean,std)
        recon=self.bvae.decoder(z)
        reconstruction=self.bvae.dae_net(recon)
        return reconstruction,z
    def symbol_from_image(self,image):
        recon,m,s=self.bvae.encoder(image)
        z=reparametrize(mean,std)
        symbol=self.decoder(z)
        return symbol,z
    def encode_to_latent(self,x):
        parametrized=self.encoder(x)
        mean=parametrized[:,:self.latent_size]
        std=parametrized[:,self.latent_size:]
        return mean,std
    def compute_loss(self,image,original,reconstruction,mu,logvar):
        batch_size = original.size(0)
        #reconstruction_loss= F.mse_loss((reconstruction,original), size_average=False).div(batch_size)
        reconstruction_loss = -(original * torch.log(reconstruction) + (1 - original) * torch.log(1 - reconstruction)).sum() / batch_size
        #print(type(mu))
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))
        klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        klds=klds.mean(0).sum()
        m,s=self.bvae.encode_to_latent(image)
        #if m.data.ndimensions() == 4:
        m = m.view(m.size(0),m.size(1))
        #if s.data.ndimension()== 4:
        s = s.view(s.size(0),s.size(1))
        #doublekld= 0.5* (-1 + s.exp() / mu.exp() + ((m - mu)**2) / s.exp() + logvar - m )
        doublekld=(0.5 * (logvar - s +
                                torch.exp(s - logvar) +
                                torch.square(m - mu) / torch.exp(logvar) -
                                1))
        #doublekld= 0.5* (-1 +  mu.exp() / m.exp() + ((mu - m)**2) / logvar + logvar - m )
        doublekld=doublekld.mean(0).sum()
        return reconstruction_loss+ self.beta * klds + self.lmbd * doublekld,doublekld
class BVAE2(nn.Module):
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
    def encode_to_latent(self,x):
        parametrized=self.encoder(x)
        mean=parametrized[:,:self.latent_size]
        std=parametrized[:,self.latent_size:]
        return mean,std
class SCAN_recomb(nn.Module):
    def __init__(self,scan):
        super().__init__()
        self.scan=scan
        self.mean_conv1=nn.Conv1d(2,512,kernel_size=1,stride=1)
        self.mean_conv2=nn.Conv1d(512,3,kernel_size=1,stride=1)
        self.std_conv1=nn.Conv1d(2,512,kernel_size=1,stride=1)
        self.std_conv2=nn.Conv1d(512,3,kernel_size=1,stride=1)
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
    def forward(self,one_hot1,one_hot2,op_vec):
        m1,s1=self.scan.encode_to_latent(one_hot1)
        m2,s2,=self.scan.encode_to_latent(one_hot2)
        m1=m1.unsqueeze(1)
        s1=s1.unsqueeze(1)
        m2=m2.unsqueeze(1)
        s2=s2.unsqueeze(1)
        means=torch.cat((m1,m2),1)
        stds=torch.cat((s1,s2),1)
        mean_conv=self.mean_conv1(means)
        mean_conv=self.relu1(mean_conv)
        mean_conv=self.mean_conv2(mean_conv)
        #print(mean_conv.shape)
        #print(op_vec.shape)
        mean_conv_pick_type=torch.mul(mean_conv,op_vec)
        mean_conv_out=torch.sum(mean_conv_pick_type,dim=1)
        std_conv=self.std_conv1(stds)
        std_conv=self.relu2(std_conv)
        std_conv=self.std_conv2(std_conv)
        std_conv_pick_type=torch.mul(std_conv,op_vec)
        std_conv_out=torch.sum(std_conv_pick_type,dim=1)
        return mean_conv_out,std_conv_out
    def compute_loss(self,image,mu,logvar,y_ground_truth):
        batch_size=image.size(0)
        m,s=self.scan.bvae.encode_to_latent(image)
        m1,s1=self.scan.encode_to_latent(y_ground_truth)
        mu = mu.view(mu.size(0), mu.size(1))
        logvar = logvar.view(logvar.size(0), logvar.size(1))
        m = m.view(m.size(0),m.size(1))
        m1 = m1.view(m1.size(0),m1.size(1))
        s1 = s1.view(s1.size(0),s1.size(1))
        s = s.view(s.size(0),s.size(1))
        doublekld2=(0.5 * (logvar - s1 +
                                torch.exp(s1 - logvar) +
                                torch.square(m1 - mu) / torch.exp(logvar) -
                                1))
        doublekld2=doublekld2.mean(0).sum()
        return doublekld2

