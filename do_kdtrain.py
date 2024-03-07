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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import glob
import os, gc
import sys
import argparse
from time import time
from datetime import datetime
from tqdm import tqdm
from natsort import natsorted
import logging
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchmetrics

from torch.utils.tensorboard import SummaryWriter

from timm.utils import accuracy

from models.make_models import make_vanilla_model
from models.vit3d import Vision_Transformer3D
from utils.optimizers import make_scheduler
from utils.utils import adjust_learning_rate_halfcosine, adjust_alpha, set_requires_grad, loop_iterable, save_model, load_checkpoint

class CrossEntropyLossForSoftTarget(nn.Module):
    def __init__(self, T=2):
        super(CrossEntropyLossForSoftTarget, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred.div(self.T)
        y_gt_soft = y_gt.div(self.T)
        return -(self.softmax(y_gt_soft)*self.logsoftmax(y_pred_soft)).mean().mul(self.T*self.T)

def do_kdtrain(cfg, args, FILENAME_POSTFIX, model, teacher_model, criterion, optimizer, scaler, source_loader, 
             source_dataset, logger, early_stopper, do_valid, 
             test_dataloader):
    """
    Do vanilla mode training

    """
    # Read batch size
    batch_size = cfg['DATALOADER']['BATCH_SIZE']

    # Calculate iter per epoch
    N_src = source_dataset.__len__()
    iter_per_epoch = source_dataset.__len__()/batch_size

    # Read epochs
    epochs = cfg['TRAINING']['EPOCHS']

    # Train the Model
    batch_time, net_time = [], []
    # steps = args.iter_start

    iter_start = args.iter_start
    steps = args.iter_start
    
    # performance metrics helpers
    train_acc, val_acc = 0, 0
    average_loss = 0
    correct = 0

    if cfg['SOLVER']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    soft_criterion = CrossEntropyLossForSoftTarget(T=args.kd_temperature)
    teacher_model.eval()
    
    for epoch in range(int(iter_start/iter_per_epoch), epochs):
        model.train()
        end = time()

        # Set the learning rate for each layer based on the decay factor
        # if cfg['TRAINING']['LAYERWISE_LR_DECAY']:
        #     for idx, param_group in enumerate(optimizer.param_groups):
        #         param_group['lr'] = optimizer.param_groups[idx]['lr'] * cfg['TRAINING']['LAYERWISE_LR_DECAY']

        for batch_data in source_loader:
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]
            
            # adjust_learning_rate_halfcosine(optimizer, steps / len(source_loader) + epoch, cfg)

            if cfg['TRAINING']['USE_GPU']:
                images, labels = (
                    batch_data["image"].cuda(non_blocking=True),
                    batch_data["label"].cuda(non_blocking=True)
                )
                if cfg['MODEL']['patch_embed_fun'][-2:] == '2d':
                    images = images.squeeze(1)
                                    
            t = time()
            
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                outputs = model(images)
                # teacher model forward
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                if args.deit_loss_type == 'hard':
                    outputs, outputs_kd = outputs
                    base_loss = criterion(outputs, labels)
                    distillation_loss = criterion(outputs_kd, teacher_outputs.argmax(dim=1))
                    loss = base_loss * (1 - args.alpha) + distillation_loss * args.alpha
                elif args.deit_loss_type == 'soft':
                    outputs, outputs_kd = outputs
                    base_loss = criterion(outputs, labels)
                    
                    distillation_loss = F.kl_div(F.log_softmax(outputs_kd / args.kd_temperature, dim=1),
                                F.softmax(teacher_outputs / args.kd_temperature, dim=1),
                                reduction='sum',
                                log_target=True) * (args.kd_temperature**2) / outputs_kd.numel()
                    
                    loss = base_loss * (1 - args.alpha) + distillation_loss * args.alpha
                    
                else:
                    loss = criterion(outputs, labels) * (1 - args.alpha)\
                        + soft_criterion(outputs, teacher_outputs) * args.alpha

            # Forward + Backward + Optimize
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
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            average_loss += float(loss.item())
            
            steps += 1
            wandb.log({"train loss": loss.item()})

            end = time()
        
        # save_model(args, cfg, model, cfg['TRAINING']['CHECKPOINT'] + FILENAME_POSTFIX, epoch, steps)
        # print(f'Saved: {cfg["TRAINING"]["CHECKPOINT"]}{FILENAME_POSTFIX}_{epoch}_{steps}')

        if do_valid:
            if args.deit_loss_type == 'hard':
                model.training = False
            test_acc, balanced_acc, test_auc, _, _ = do_inference(cfg, args, model, test_dataloader, logger, is_validation=True)
            if args.deit_loss_type == 'hard':
                model.training = True
            model.train()
            if val_acc < test_acc:
                val_acc = test_acc
                # saving the best model by first removing previous best models (for saving memory)
                save_model(args, cfg, model, cfg['TRAINING']['CHECKPOINT'] + 'BEST_MODEL_' + FILENAME_POSTFIX, epoch, steps)
                print(f'Best model saved: {cfg["TRAINING"]["CHECKPOINT"]}{FILENAME_POSTFIX}_{epoch}_{steps}')

        # Printing train stats and logging
        print(f'[{epoch+1}/{epochs}] {steps}) LR {lr:.7f}, Loss: {average_loss/iter_per_epoch:.3f}, Acc {100*correct/N_src:.2f}%, N {N_src}, Test Acc {test_acc:.2f}%, Bal acc {balanced_acc:.2f}%')
        logger.info(f'[{epoch+1}/{epochs}] {steps}) LR {lr:.7f}, Loss: {average_loss/iter_per_epoch:.3f}, Acc {100*correct/N_src:.2f}%, N {N_src}, Test Acc {test_acc:.2f}%, Bal acc {balanced_acc:.2f}%')
        
        torch.cuda.empty_cache()

        wandb.log({"lr": lr, "epoch": epoch+1, "avg loss": average_loss/iter_per_epoch, 
                   "train acc": 100*correct/N_src, "test acc": test_acc, "bal acc": balanced_acc, "test auc": test_auc})
        average_loss, correct = 0, 0
                
        if epoch > 20 and cfg['TRAINING']['EARLY_STOPPING']:
            early_stopper(test_acc)
            if early_stopper.early_stop:
                print('Early stopping and saving the model...')
                # saving the last model by first removing previous models (for saving memory)
                break
        
        if cfg['SOLVER']['scheduler'] == 'cosine':                
            scheduler.step()
        
        if os.path.exists(cfg['TRAINING']['CHECKPOINT']+'/stop.txt'):
            # break without using CTRL+C
            # just create stop.txt file in cfg['TRAINING']['CHECKPOINT']
            break
        
        if os.path.exists(cfg['TRAINING']['CHECKPOINT']+'/pdb.txt'):
            import pdb; pdb.set_trace()
    
    # saving the last model by first removing previous models (for saving memory)
    # save_model(args, cfg, model, cfg['TRAINING']['CHECKPOINT']+FILENAME_POSTFIX, epoch, steps)
    # print(f'Saved Last Model As: {cfg["TRAINING"]["CHECKPOINT"]}{FILENAME_POSTFIX}_{epoch}_{steps}')
    
    # Load the best-performing model
    model = load_checkpoint(args, cfg, model, 'BEST_MODEL_' + FILENAME_POSTFIX)

    return model

@torch.no_grad()
def do_inference(cfg, args, model, test_loader, logger, is_validation=True):
    """

    Parameters
    ----------
    cfg : config yaml
        Config read in yaml format file
    args : argument parser
        Arguments red from command line
    test_loader : torch.Dataloader
        test dataset loader

    Returns
    -------
    acc : float
        accuracy of inference
    """
    
    corrects = 0
    auroc_calc = torchmetrics.AUROC(task="multiclass", num_classes=len(args.classes_to_use))
    # calibration_calc = torchmetrics.CalibrationError(task='multiclass', num_classes=len(args.classes_to_use))
    
    logits_arr, preds_arr, labels_arr = [], [], []
    n_datapoints = 0
    model.eval()
    for batch_data in test_loader:

        if cfg['TRAINING']['USE_GPU']:
            images = batch_data["image"].cuda(non_blocking=True)
            if cfg['MODEL']['patch_embed_fun'][-2:] == '2d':
                    images = images.squeeze(1)
        labels = batch_data["label"] # (1)
        
        outputs = model(images) # (1,2))
        logits = F.softmax(outputs, dim=-1).cpu().data # (1,2)

        logits_arr.append(logits.view(-1).detach().cpu().tolist())
        labels_arr.append(labels.view(-1).detach().cpu().tolist())
        
        preds = torch.argmax(logits, dim=1)
        corrects += torch.sum(preds == labels)
        n_datapoints += outputs.shape[0]

        preds_arr.append(preds.view(-1).detach().cpu().tolist())
    
    test_acc = corrects.item()/n_datapoints
    # test_auc = roc_auc_score(target_auc.numpy(), output_auc.numpy(), average='weighted', multi_class='ovo')
    test_auc = auroc_calc(torch.tensor(logits_arr), torch.tensor(labels_arr).squeeze()).item()
    # test_cal = calibration_calc(torch.tensor(logits_arr), torch.tensor(labels_arr).squeeze()).item()
    
    balanced_acc = balanced_accuracy_score(labels_arr, preds_arr)
    
    logger.info(f'TESTING: Number of datapoints: {n_datapoints}), Accuracy {100*test_acc:.2f}%, Bal ACC {100*balanced_acc:.2f}%, AUC {100*test_auc:.2f}%')

    if not is_validation:
        print(f'TESTING: Number of datapoints: {n_datapoints}), Accuracy {100*test_acc:.2f}%, Bal ACC {100*balanced_acc:.2f}%, AUC {100*test_auc:.2f}%')

    del logits_arr, labels_arr, preds_arr, auroc_calc
    gc.collect()

    return 100*test_acc, 100*balanced_acc, 100*test_auc, corrects.item(), n_datapoints
