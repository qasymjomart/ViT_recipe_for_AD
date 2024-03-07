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

from dataloaders.make_dataloaders import make_train_test_split_of_target, make_monai_dataloaders
from models.make_models import make_vanilla_model
from do_train import do_train, do_inference
from utils.utils import EarlyStopping, load_pretrained_checkpoint
from utils.optimizers import make_optimizer

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
        "source dataset": args.source,
        "target dataset": args.target,
        "pre-trained model": args.use_pretrained if args.use_pretrained else None,
        "lr": cfg['SOLVER']['lr'],
        "checkpoint_type": cfg['TRAINING']['CHECKPOINT_TYPE'],
        "dir": "./wandb/"+FILENAME_POSTFIX+"/"
        }
    )

def setup_logger(ILENAME_POSTFIX, timestamp_current):
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
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Name of the config file')
    parser.add_argument('--savename', type=str, help='Experiment name (used for saving files)')
    parser.add_argument('--classes_to_use', nargs='+', type=str, help='Classes to use (enter by separating by space, e.g. CN AD MCI)')
    parser.add_argument('--mode', required=True, type=str, help='Experiment mode type (vanilla, uda)')
    parser.add_argument('--source', type=str, help='Source dataset')
    parser.add_argument('--target', type=str, help='Target dataset')
    parser.add_argument('--seed', type=int, help='Experiment seed (for reproducible results)')
    parser.add_argument('--devices', type=str, help='GPU devices to use')
    parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count of training')
    parser.add_argument('--patch_embed_fun', type=str, help='Patch embed function to use')
    parser.add_argument('--use_pretrained', type=str, help='Path to pre-trained model checkpoint to load')
    parser.add_argument('--checkpoint', default='./checkpoints/', type=str, help='Checkpoint model path')
    args = parser.parse_args()

    # Loads config file for fixed configs
    f_config = open(args.config_file,'rb')
    cfg = yaml.load(f_config, Loader=yaml.FullLoader)

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

    # Set up logger file
    logger = setup_logger(FILENAME_POSTFIX, timestamp_current)
    logger.setLevel(logging.DEBUG)
    logger.info('Process number: %d'%(os.getpid()))
    logger.info("Started training. Savename : " + args.savename + " " + args.mode)
    logger.info("Seed : " + str(args.seed))
    logger.info("Source dataset : " + args.source + ", Target dataset : " + args.target)
    logger.info("Training mode (vanilla/uda/etc) : " + args.mode)
    
    # Monai logs foldernames
    cfg['TRANSFORMS']['cache_dir_train'] = './monai_logs/train_' + FILENAME_POSTFIX
    cfg['TRANSFORMS']['cache_dir_test'] = './monai_logs/test_' + FILENAME_POSTFIX

    train_test_split_paths = make_train_test_split_of_target(cfg, args, test_size=0.2)

    
    if args.mode == 'source_only':
        train_dataloader, test_dataloader, train_dataset, test_dataset, train_datalist, test_datalist = make_monai_dataloaders(cfg, args, train_test_split_paths)
    
    cfg['MODEL']['patch_embed_fun'] = args.patch_embed_fun
    # Number of classes to use
    cfg['MODEL']['n_classes'] = len(args.classes_to_use)
    print('-----------------------------')
    print('Number of classes to be used: ', args.classes_to_use, cfg['MODEL']['n_classes'])
    print('-----------------------------')

    if args.mode == 'vanilla' or args.mode == 'source_only':
        model = make_vanilla_model(cfg, args)
    elif args.mode == 'uda':
        model = make_uda_model(cfg, args)
    
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
    
    # Initialize loss function and optimizer
    weight_balance = torch.Tensor(list(train_datalist.__getlabelsratio__().values())).cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_balance)
    
    if len([x for x in args.devices.split(",")]) > 1: # if more than 1 GPU selected
        model = torch.nn.DataParallel(model)
    
    # Early Stopper
    early_stopper = None
    if cfg['TRAINING']['EARLY_STOPPING']:
        early_stopper = EarlyStopping(patience=cfg['TRAINING']['EARLY_STOPPING_PATIENCE'], 
                                      min_delta=cfg['TRAINING']['EARLY_STOPPING_DELTA'])        
    
    # Save all configs and args just in case
    logger.info(cfg)
    logger.info(args)

    # Init wandb
    wandb_setup(cfg, args, FILENAME_POSTFIX)

    trained_model = do_train(
        cfg, args, FILENAME_POSTFIX, model, criterion, optimizer, scaler, train_dataloader,
        train_dataset, logger, early_stopper, True,
        test_dataloader
    )
    

    test_acc, bal_acc, corrects, n_datapoints = do_inference(
            cfg, args, trained_model, test_dataloader, logger
        )




    






