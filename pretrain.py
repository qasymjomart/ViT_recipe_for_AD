# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
# from skimage import io, color, segmentation

import os
import glob
from datetime import datetime
import argparse
import random
import yaml
import logging
import wandb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda import amp
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from dataloaders.make_dataloaders import make_mae_pretraining_dataloaders
from models.make_models import make_mae_model
from do_pretrain import do_pretrain
from utils.utils import EarlyStopping, load_pretrained_checkpoint
from utils.optimizers import make_optimizer
from utils.loss import PrototypePseudoLabeling

# Set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Seed is set.')

def setup_tensorboard(FILENAME_POSTFIX, timestamp_current):
    # Set up the Tensorboard log
    if len(glob.glob('./runs/runs_' + FILENAME_POSTFIX + '*')) > 0:
        print('Tensorboard log dir found...')
        writer = SummaryWriter(log_dir= glob.glob('./runs/runs_' + FILENAME_POSTFIX + '*')[-1])
    else:
        writer = SummaryWriter(log_dir='./runs/runs_' + FILENAME_POSTFIX + '_' + timestamp_current)
    
    return writer

def wandb_setup(cfg, args, FILENAME_POSTFIX):
    # start a new wandb run to track this script
    os.makedirs('./wandb/'+FILENAME_POSTFIX+'/', exist_ok=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="MAE Pre-training",
        name=FILENAME_POSTFIX,
        
        # track hyperparameters and run metadata
        config={
        "config_file": args.config_file,
        "datasets_to_use": args.datasets_to_use,
        "lr": cfg['SOLVER']['lr'],
        "dir": "./wandb/"+FILENAME_POSTFIX+"/",
        "mask ratio": args.mask_ratio,
        "seed": args.seed,
        "use_aug": args.use_aug
        }
    )

def setup_logger(FILENAME_POSTFIX, timestamp_current):
    # Set up the logger
    if len(glob.glob('./logs/' + FILENAME_POSTFIX + '*')) > 0:
        print('Logger found...')
        print('----------------')
        logging.basicConfig(filename=glob.glob('./logs/' + FILENAME_POSTFIX + '*')[-1],
                            format='%(asctime)s %(message)s',
                            filemode='a',
                            level=logging.DEBUG, 
                            force=True)
    else:
        logging.basicConfig(filename='./logs/' + FILENAME_POSTFIX + '_' + timestamp_current + '.log', 
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        level=logging.DEBUG, 
                        force=True)
    logger = logging.getLogger()
    return logger

if __name__ == '__main__':

    # Parse some variable configs
    parser = argparse.ArgumentParser(description='Train UDA model for MRI imaging for classification of AD')
    parser.add_argument('--config_file', type=str, default='config_pretrain.yaml', help='Name of the config file')
    parser.add_argument('--model', default='mae', type=str, help='Pre-training model')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--datasets_to_use', nargs='+', type=str, help='Datasets to use for pre-training MAE')
    parser.add_argument('--mode', required=True, type=str, default='pretrain', help='Experiment mode type (vanilla, uda)')
    parser.add_argument('--seed', type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Mask ratio used for MAE')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--use_pretrained', type=str, help='Path to pre-trained model checkpoint to load')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count of training')
    # parser.add_argument('--checkpoint', default='./checkpoints/', type=str, help='Checkpoint model path')
    args = parser.parse_args()

    # Loads config file for fixed configs
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader)
    # print(torch.cuda.device_count())
    # Set seed
    set_seed(args.seed)

    # Set up GPU devices to use
    if cfg['TRAINING']['USE_GPU']:
        print(('Using GPU %s'%args.devices))
        os.environ["CUDA_DEVICE_ORDER"]=cfg['TRAINING']['CUDA_DEVICE_ORDER']
        os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
    else:
        print('CPU mode')
    print('Process number: %d'%(os.getpid()))
    print('-----------------------------')

    FILENAME_POSTFIX = args.savename + '_' + args.mode + '_seed_' + str(args.seed)

    timestamp_current = datetime.now()
    timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")

    # Set up the Tensorboard log
    # writer = setup_tensorboard(FILENAME_POSTFIX, timestamp_current)
    # Set up logger file
    logger = setup_logger(FILENAME_POSTFIX, timestamp_current)
    logger.setLevel(logging.DEBUG)
    logger.info('Process number: %d'%(os.getpid()))
    logger.info("Started training. Savename : " + args.savename + " " + args.mode)
    logger.info("Seed : " + str(args.seed))
    logger.info("Training mode : " + args.mode)

    # Init wandb
    wandb_setup(cfg, args, FILENAME_POSTFIX)
    
    # Monai logs foldernames
    cfg['TRANSFORMS']['cache_dir_train'] = './monai_logs/train_' + FILENAME_POSTFIX
    
    if args.model == 'mae':
        model = make_mae_model(cfg, args)
        print(args.model, " is built.")
        pretraining_dataloader, pretrain_dataset = make_mae_pretraining_dataloaders(cfg, args)
    
    # Load pre-trained model weights (MAE)
    if args.use_pretrained is not None:
        model = load_pretrained_checkpoint(model, args.use_pretrained, cfg['TRAINING']['CHECKPOINT_TYPE'])
        
    # Move model to GPU
    if cfg['TRAINING']['USE_GPU']:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Num of parameters in the model: ', params)
        model.cuda()

    optimizer = make_optimizer(cfg, args, model)
    scaler = amp.GradScaler()
    
    # Split the devices string into a list of integers
    device_ids = list(map(int, args.devices.split(",")))
    # Check if more than one GPU is selected
    if len(device_ids) > 1:
        print(f"Multi-GPU training on devices: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(f"Model is on device: {next(model.parameters()).device}")
    
    # Save all configs and args (just in case)
    logger.info(cfg)
    logger.info(args)

    do_pretrain(cfg=cfg,
                args=args,
                FILENAME_POSTFIX=FILENAME_POSTFIX,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                pretrain_loader=pretraining_dataloader,
                pretrain_dataset=pretrain_dataset,
                logger=logger
                )

