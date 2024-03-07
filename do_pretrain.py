# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import math
from PIL import Image
from skimage import io, color, segmentation

import glob, os
from time import time
from natsort import natsorted
import logging
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from utils.utils import adjust_learning_rate_halfcosine, save_model

def do_pretrain(cfg, args, FILENAME_POSTFIX, model, optimizer, scaler, pretrain_loader, pretrain_dataset, logger):
    """
    Do MAE pre-training

    """
    # Read log period
    log_period = cfg['TRAINING']['LOGGING_PERIOD']
    logger.info('Started training')

    # Read evalutation period

    # Read batch size
    batch_size = cfg['DATALOADER']['BATCH_SIZE']

    # Calculate iter per epoch
    iter_per_epoch = pretrain_dataset.__len__()/batch_size

    # Read epochs
    epochs = cfg['TRAINING']['EPOCHS']

    # Train the Model
    batch_time, net_time = [], []

    iter_start = args.iter_start
    steps = args.iter_start
    
    # performance metrics helpers
    average_loss = 0

    for epoch in range(int(iter_start/iter_per_epoch), epochs):
        model.train()
        end = time()

        for batch_data in pretrain_loader:
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]

            adjust_learning_rate_halfcosine(optimizer, steps / len(pretrain_loader) + epoch, cfg)

            if cfg['TRAINING']['USE_GPU']:
                optimizer.zero_grad(set_to_none=True)
                t = time()
                if args.model == 'mae':
                    images = batch_data["image"].cuda(non_blocking=True)   
                    # with amp.autocast(enabled=True):
                    loss, _, _ = model(images, mask_ratio=args.mask_ratio)
                    # print('loss:', loss)
                elif args.model == 'simmim':
                    images, masks = batch_data
                    masks = torch.unsqueeze(masks, 1).cuda(non_blocking=True)
                    inputs = images["image"].cuda(non_blocking=True)
                    with amp.autocast(enabled=True):
                        loss = model(inputs, masks)
                    
            # Forward + Backward + Optimize
            if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
                loss = loss.sum()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            lr = optimizer.param_groups[0]["lr"]

            torch.cuda.synchronize()

            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]

            # other way to calculate accuracy
            average_loss += float(loss.item())
            # writer.add_scalar(FILENAME_POSTFIX + '_loss', float(loss.item()), steps)

            steps += 1
            wandb.log({"pretrain train loss": loss.item()})
            
            end = time()
        
        print(('[%2d/%2d] %5d) [batch load % 2.3fs, net %1.2fs], LR %.6f, Loss: % 1.3f, N %3d' %(
                epoch+1, epochs, steps, np.mean(batch_time), np.mean(net_time), lr, average_loss/iter_per_epoch, iter_per_epoch*batch_size)))
        logger.info('[%2d/%2d] %5d) [batch load % 2.3fs, net %1.2fs], LR %.6f, Loss: % 1.3f, N %3d' %(
                    epoch+1, epochs, steps, np.mean(batch_time), np.mean(net_time), lr, average_loss/iter_per_epoch, iter_per_epoch*batch_size))
        average_loss = 0
        # scheduler.step()
        save_model(args, cfg, model, cfg['TRAINING']['CHECKPOINT'] + FILENAME_POSTFIX, epoch, steps)
        print('Saved: ' + cfg['TRAINING']['CHECKPOINT'] + FILENAME_POSTFIX + '_' + str(epoch) + '_' + str(steps))

        if os.path.exists(cfg['TRAINING']['CHECKPOINT']+'/stop.txt'):
            # break without using CTRL+C
            # just create stop.txt file in cfg['TRAINING']['CHECKPOINT']
            break
        
        if os.path.exists(cfg['TRAINING']['CHECKPOINT']+'/pdb.txt'):
            import pdb; pdb.set_trace()

    # saving the last model by first removing previous models (for saving memory)
    save_model(args, cfg, model, cfg['TRAINING']['CHECKPOINT']+ FILENAME_POSTFIX, epoch, steps)
    print('Saved Last Model As: ' + cfg['TRAINING']['CHECKPOINT'] + FILENAME_POSTFIX + '_' + str(epoch) + '_' + str(steps))