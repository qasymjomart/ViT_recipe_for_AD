# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from time import time
from datetime import datetime
from tqdm import tqdm
from natsort import natsorted

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from timm.scheduler import CosineLRScheduler

_optimizers_factory = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'RMSprop': torch.optim.RMSprop
}

def make_optimizer(cfg, args, model):
    """
    Initialize an optimizer based on the configs of cfg and args

    Parameters:
    -----------
    cfg, args: config and argument parser from the command line
    model: torch.nn

    Returns:
    ---------
    optimizer: torch.optim.SGD

    """
    optimizer_type = cfg['SOLVER']['optimizer']
    if optimizer_type == 'SGD':
        optimizer = _optimizers_factory[optimizer_type](
            model.parameters(),
            lr=cfg['SOLVER']['lr'],
            weight_decay=cfg['SOLVER']['weight_decay'],
            )
    elif optimizer_type == 'RMSprop':
        optimizer = _optimizers_factory[optimizer_type](
            model.parameters(),
            lr=cfg['SOLVER']['lr'],
            weight_decay=cfg['SOLVER']['weight_decay'],
            alpha=cfg['SOLVER']['alpha'],  # RMSprop-specific parameter
            momentum=cfg['SOLVER']['momentum']  # RMSprop-specific parameter
        )
    else:
        optimizer = _optimizers_factory[optimizer_type](
            model.parameters(),
            lr=cfg['SOLVER']['lr'],
            weight_decay=cfg['SOLVER']['weight_decay'],
            betas=(cfg['SOLVER']['beta1'], cfg['SOLVER']['beta2'])
            )
    
    return optimizer

def make_pretrain_optimizer(cfg, args, model):
    """
    Initialize an optimizer based on the configs of cfg and args

    Parameters:
    -----------
    cfg, args: config and argument parser from the command line
    model: torch.nn

    Returns:
    ---------
    optimizer: torch.optim.SGD

    """
    optimizer = _optimizers_factory[cfg['SOLVER']['optimizer']](
                                model.parameters(),
                                lr=cfg['SOLVER']['lr'],
                                weight_decay=cfg['SOLVER']['weight_decay'],
                                betas=(cfg['SOLVER']['beta1'], cfg['SOLVER']['beta2'])
                                )
    
    return optimizer

def make_scheduler(cfg, args, optimizer):
    """Make a scheduler

    Parameters
    ----------
    cfg, args: config and argument parser from the command line
    optimizer : optimizer to wrap into the scheduler
    """
    if cfg['SOLVER']['scheduler'] == 'cosine':        
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg['SOLVER']['t_initial'],
            lr_min=cfg['SOLVER']['min_lr'],
            t_in_epochs=True,
            cycle_decay=cfg['SOLVER']['cycle_decay'],
            cycle_limit=cfg['SOLVER']['cycle_limit'],
            warmup_lr_init=cfg['SOLVER']['warmup_lr'],
            warmup_t=cfg['SOLVER']['warmup_epochs']
        )
    
    return lr_scheduler

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return 
