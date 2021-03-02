DATA_PATH="data"
UNNORM_DATA_PATH="unnormalized.npy"
NORM_DATA_PATH="normalized.npy"
PRETRAIN=True
DATA_CHANNEL=3
DAE_ENCODER_SIZES=[32,32,64,64]
DAE_DECODER_SIZES=[64,64,32,32]
BVAE_ENCODER_SIZES=[32,32,64,64]
BVAE_DECODER_SIZES=[64,64,32,32]
DAE_ENCODER_ACTIVATION=[True,True,True,True]
DAE_DECODER_ACTIVATION=[True,True,True,False]
BVAE_ENCODER_ACTIVATION=[True,True,True,True]
BVAE_DECODER_ACTIVATION=[True,True,True,False]
DAE_BOTTLENECKS_SIZE=100
DAE_lr=0.001
DAE_eps=0.00000001
DAE_PRETRAIN=True
BVAE_PRETRAIN=True
TRAIN_SIZE=10000
generator_params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 3
          }
scan_generator_params={
    "batch_size" : 16,
    "num_workers" : 3,
    "shuffle" : True,
    "pin_memory" : True
}
recomb_generator_params={'batch_size': 16, 
                    'num_workers': 3, 
                    'pin_memory': True, 
                    'shuffle': True}

SCAN_CHECKPOINT="SCAN_LOG"
RECOMB_CHECKPOINT="RECOMB_LOG"
DAE_CHECKPOINT="DAE_LOG"
BVAE_CHECKPOINT="BVAE_LOG"
DAE_LOG=DAE_CHECKPOINT+"/DAE_LOG.txt"
BVAE_LOG=BVAE_CHECKPOINT+"/BVAE_LOG.txt"
DAE_TRAIN_EPOCH=2000
DAE_LOAD_PATH=DAE_CHECKPOINT+"/"+str(DAE_TRAIN_EPOCH)+".pt"
BVAE_BETA=53
LATENT_DIM=32
BVAE_EPS=0.00000001
BVAE_LR=0.0001
BVAE_TRAIN_EPOCH=2000
BVAE_CHECKPOINT="BVAE_LOG"
BVAE_LOAD_PATH=BVAE_CHECKPOINT+"/"+str(BVAE_TRAIN_EPOCH)+".pt"
VIS_RECON_PATH="VIS_RECON"
VIS_LATENT_TRAVERSAL="LATENT_TRAVERSAL"
object_type_dict = {
    "undefined": -1,
    "lollypop": 0,
    "top hat": 1,
    "briefcase": 2
}
operator_type_dict = {
    "AND": 0,
    "INCOMMON": 1,
    "DIFFERENCE": 2,
}
color_dict = {
    "undefined": -1,
    "white" : 0,
    "blue" : 1,
    "dark_purple": 2,
    "light_blue" :3,
    "green":4,
    "light_purple":5,
    "lime":6,
    "grey":7,
    "brown":8,
    "dark_pink":9,
    "red":10,
    "light_pink":11,
    "dark_yellow":12,
    "bright_pink":13,
    "bright_yellow":14,
    "open_yellow":15

}
colours=[(x,color_dict[x]) for x in color_dict]
object_types=[(x,object_type_dict[x]) for x in object_type_dict]
ops=[(x,operator_type_dict[x]) for x in operator_type_dict]
