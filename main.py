from model.py import DAE,BVAE,reparametrize
import dataset.py as dataset
import utils.py as utils
import train.py as train
import configs.py as config 
import cv2
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

###INITIALIZE PATHS###
try:
    os.mkdir(config.DAE_CHECKPOINT)
    os.mkdir(config.BVAE_CHECKPOINT)
    os.mkdir(config.VIS_RECON_PATH)
    os.mkdir(config.VIS_LATENT_TRAVERSAL)


#Read RGB data from data folder and save
dataset.read_data(config.DATA_PATH,config.UNNORM_DATA_PATH)
#Turn data into hsv and normalize to mean=0 std=1
channel_mean,channel_std=dataset.normalize_data(NORM_DATA_PATH,UNNORM_DATA_PATH)
#load into array
data=dataset.load_data(NORM_DATA_PATH)

train_data,test_data=train.split_train_test(data,config.TRAIN_SIZE)

writer = SummaryWriter()

DAE_net=DAE(config.DATA_CHANNEL,config.DAE_ENCODER_SIZES,config.DAE_DECODER_SIZES,config.DAE_ENCODER_ACTIVATION,config.DAE_DECODER_ACTIVATION,config.DAE_config.DAE_BOTTLENECK_SIZE,"bernouilli")

optim_dae= torch.optim.Adam(DAE_net.parameters(),lr=config.DAE_lr,eps=config.DAE_eps)

if not config.DAE_PRETRAIN:
        train_set=train.DAEdata(train_data)
        
        dae_training_generator=data.DataLoader(train_set,**config.generator_params)
        
        test_set=train.DAEdata(test_data)
        
        dae_test_generator=data.DataLoader(test_set,**config.generator_params)
        
        train.train_dae(DAE_net,optim_dae,dae_training_generator,dae_test_generator,nn.MSELoss(),config.DAE_CHECKPOINT,config.DAE_TRAIN_EPOCH,writer,config.DAE_LOG):
else:
        utils.load_model(config.DAE_LOAD_PATH,DAE_net,optim_dae)
        
        
BVAE_net=BVAE(config.DATA_CHANNEL,config.BVAE_ENCODER_SIZES,config.BVAE_DECODER_SIZES,config.BVAE_ENCODER_ACTIVATION,BVAE_DECODER_ACTIVATION,config.LATENT_DIM,config.BVAE_BETA,DAE_net)

optim_bvae = torch.optim.Adam(list(BVAE_net.encoder.parameters())+list(BVAE_net.decoder.parameters()),lr=config.BVAE_LR,eps=config.BVAE_EPS)

if not config.BVAE_PRETRAIN:
        train_set=train.VAEdata(train_data)
        
        vae_training_generator=data.DataLoader(train_set,**config.generator_params)
        
        test_set=train.VAEdata(test_data)
        
        vae_test_generator=data.DataLoader(test_set,**config.generator_params)
        
        train.train_bvae(BVAE_net,optim_bvae,vae_training_generator,vae_test_generator,config.BVAE_CHECKPOINT,config.BVAE_TRAIN_EPOCH,writer,config.BVAE_LOG)
else:
        utils.load_model(config.BVAE_LOAD_PATH,BVAE_net,optim_bvae)
        
for batch_id,batch in enumerate(0,vae_test_generator):
        utils.visualize_recon(BVAE_net,DAE_net,channel_mean,channel_std,config.VIS_RECON_PATH)
        utils.latent_traversal(BVAE_net,DAE_net,channel_mean,channel_std,config.VIS_LATENT_TRAVERSAL)
        
        
