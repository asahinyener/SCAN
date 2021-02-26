from model import DAE,BVAE,SCAN,BVAE2,SCAN_recomb
import dataset
import dataloader
import utils
import train
import configs as config
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
    os.mkdir(config.SCAN_CHECKPOINT)
    os.mkdir(config.RECOMB_CHECKPOINT)
    os.mkdir(config.VIS_RECON_PATH)
    os.mkdir(config.VIS_LATENT_TRAVERSAL)


#Read RGB data from data folder and save
dataset.read_data(config.DATA_PATH,config.UNNORM_DATA_PATH)
#Turn data into hsv and normalize to mean=0 std=1
channel_mean,channel_std=dataset.normalize_data(NORM_DATA_PATH,UNNORM_DATA_PATH)
#load into array
data_set=dataset.load_data(NORM_DATA_PATH)
#create one hot array 
one_hots=dataloader.generate_one_hots(data_set.shape[0])
#create shuffle indexes
perm=dataset.index_generate_random(data_set)
#shuffle the data
data_set=data_set[perm]
one_hots=one_hots[perm]

train_data,test_data=dataset.split_train_test(data_set,config.TRAIN_SIZE)
oh_train_data,oh_test_data=dataset.split_train_test(one_hots,config.TRAIN_SIZE)

writer = SummaryWriter()

DAE_net=DAE(config.DATA_CHANNEL,config.DAE_ENCODER_SIZES,config.DAE_DECODER_SIZES,config.DAE_ENCODER_ACTIVATION,config.DAE_DECODER_ACTIVATION,config.DAE_config.DAE_BOTTLENECK_SIZE,"bernouilli")
DAE_net.cuda()
optim_dae= torch.optim.Adam(DAE_net.parameters(),lr=config.DAE_lr,eps=config.DAE_eps)

if not config.DAE_PRETRAIN:
        train_set=dataloader.DAEdata(train_data)
        
        dae_training_generator=data.DataLoader(train_set,**config.generator_params)
        
        test_set=dataloader.DAEdata(test_data)
        
        dae_test_generator=data.DataLoader(test_set,**config.generator_params)
        
        train.train_dae(DAE_net,optim_dae,dae_training_generator,dae_test_generator,nn.MSELoss(),config.DAE_CHECKPOINT,config.DAE_TRAIN_EPOCH,writer,config.DAE_LOG):
else:
        utils.load_model(config.DAE_LOAD_PATH,DAE_net,optim_dae)
        
        
BVAE_net=BVAE2(config.DATA_CHANNEL,config.BVAE_ENCODER_SIZES,config.BVAE_DECODER_SIZES,config.BVAE_ENCODER_ACTIVATION,BVAE_DECODER_ACTIVATION,config.LATENT_DIM,config.BVAE_BETA,DAE_net)
BVAE_net.cuda()
optim_bvae = torch.optim.Adam(list(BVAE_net.encoder.parameters())+list(BVAE_net.decoder.parameters()),lr=config.BVAE_LR,eps=config.BVAE_EPS)

if not config.BVAE_PRETRAIN:
        train_set=dataloader.VAEdata(train_data)
        
        vae_training_generator=data.DataLoader(train_set,**config.generator_params)
        
        test_set=dataloader.VAEdata(test_data)
        
        vae_test_generator=data.DataLoader(test_set,**config.generator_params)
        
        train.train_bvae(BVAE_net,optim_bvae,vae_training_generator,vae_test_generator,config.BVAE_CHECKPOINT,config.BVAE_TRAIN_EPOCH,writer,config.BVAE_LOG)
else:
        utils.load_model(config.BVAE_LOAD_PATH,BVAE_net,optim_bvae)
        
"""for batch_id,batch in enumerate(0,vae_test_generator):
        utils.visualize_recon(BVAE_net,DAE_net,channel_mean,channel_std,config.VIS_RECON_PATH)
        utils.latent_traversal(BVAE_net,DAE_net,channel_mean,channel_std,config.VIS_LATENT_TRAVERSAL)
"""





SCAN_net=SCAN(51,100,32,1,10,BVAE_net)
SCAN_net.cuda()
optim_scan = torch.optim.Adam(list(SCAN_net.encoder.parameters())+list(SCAN_net.decoder.parameters()),lr=1e-4)

oh_train_set=dataloader.SCANdata(one_hots_train,train_data)

oh_training_generator=data.DataLoader(oh_train_set,**config.scan_generator_params)

oh_test_set=dataloader.SCANdata(one_hots_test,test_data)

oh_test_generator=data.DataLoader(oh_test_set,**config.scan_generator_params)

train.train_scan(SCAN_net,optim_scan,oh_training_generator,oh_test_generator,config.SCAN_CHECKPOINT,10,writer,"output_file_path")

recomb_train_set=dataloader.RECOMBdata(data_set,perm,[15,15,15,2],20000)

recomb_training_generator=data.DataLoader(recomb_train_set,**config.recomb_generator_params)

recomb=SCAN_recomb(SCAN_net)
recomb.cuda()
optim_recomb=torch.optim.Adam(list(recomb.mean_conv1.parameters())+list(recomb.std_conv1.parameters())+list(recomb.std_conv2.parameters())+list(recomb.mean_conv2.parameters()))

train_recomb(recomb,optim_recomb,recomb_training_generator,check_point_dir):
