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

import glob
import os
import sys
import argparse
import random
import copy
import json
from tqdm import tqdm
from natsort import natsorted
import yaml
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import monai

# from make_dataloaders import make_dataloaders
from .vit3d import Vision_Transformer3D, DeiT_Transformer3D
from .maskedautoencoder3d import MaskedAutoencoderViT3D
from .maskedautoencoder3d import MaskedAutoencoderViT3D, MaskedAutoencoderDeiT3D

_models_factory = {
    'ViT3D_vanilla': Vision_Transformer3D,
    'ViT3D_linearprob': Vision_Transformer3D,
    'ViT3D_monai': monai.networks.nets.ViT,
    'ViT3D_finetune': Vision_Transformer3D,
    'ResNet152': monai.networks.nets.resnet152,
    'DeiT3D_vanilla': DeiT_Transformer3D,
    'DeiT3D_finetune': DeiT_Transformer3D,
    'MaskedAutoencoderViT3D': MaskedAutoencoderViT3D,
    'MaskedAutoencoderDeiT3D': MaskedAutoencoderDeiT3D,
}


def make_vanilla_model(cfg, args):
    """
    Make vanilla training mode Vision Transformer

    """
    if cfg['MODEL']['TYPE'] in ['ViT3D']:
        assert cfg['MODEL']['TYPE'] + '_' + args.mode in _models_factory.keys(), cfg['MODEL']['TYPE'] + '_' + args.mode + ' not in the model factory list'

        if args.vit_size in ['small']:
            cfg['MODEL']['embed_dim'] = 384
            cfg['MODEL']['depth'] = 12
            cfg['MODEL']['n_heads'] = 6
        elif args.vit_size in ['large']:
            cfg['MODEL']['embed_dim'] = 1024
            cfg['MODEL']['depth'] = 24
            cfg['MODEL']['n_heads'] = 16
        elif args.vit_size in ['huge']:
            cfg['MODEL']['embed_dim'] = 1280
            cfg['MODEL']['depth'] = 32
            cfg['MODEL']['n_heads'] = 16

        model = _models_factory[cfg['MODEL']['TYPE'] + '_' + args.mode](
            img_size        = cfg['MODEL']['img_size'],
            patch_size      = cfg['MODEL']['patch_size'],
            in_chans        = cfg['MODEL']['in_chans'],
            n_classes       = cfg['MODEL']['n_classes'],
            embed_dim       = cfg['MODEL']['embed_dim'],
            depth           = cfg['MODEL']['depth'],
            n_heads         = cfg['MODEL']['n_heads'],
            mlp_ratio       = cfg['MODEL']['mlp_ratio'],
            qkv_bias        = cfg['MODEL']['qkv_bias'],
            drop_path_rate  = cfg['MODEL']['drop_path_rate'],
            p               = cfg['MODEL']['p'],
            attn_p          = cfg['MODEL']['attn_p'],
            global_avg_pool = cfg['MODEL']['global_avg_pool'],
            patch_embed_fun = cfg['MODEL']['patch_embed_fun'],
            pos_embed_type  = cfg['MODEL']['pos_embed_type']
        )
    
    elif cfg['MODEL']['TYPE'] in ['DeiT3D']:
        assert cfg['MODEL']['TYPE'] + '_' + args.mode in _models_factory.keys(), cfg['MODEL']['TYPE'] + '_' + args.mode + ' not in the model factory list'

        if args.vit_size in ['small']:
            cfg['MODEL']['embed_dim'] = 384
            cfg['MODEL']['depth'] = 12
            cfg['MODEL']['n_heads'] = 6
        elif args.vit_size in ['large']:
            cfg['MODEL']['embed_dim'] = 1024
            cfg['MODEL']['depth'] = 24
            cfg['MODEL']['n_heads'] = 16
        elif args.vit_size in ['huge']:
            cfg['MODEL']['embed_dim'] = 1280
            cfg['MODEL']['depth'] = 32
            cfg['MODEL']['n_heads'] = 16

        model = _models_factory[cfg['MODEL']['TYPE'] + '_' + args.mode](
            img_size        = cfg['MODEL']['img_size'],
            patch_size      = cfg['MODEL']['patch_size'],
            n_classes       = cfg['MODEL']['n_classes'],
            embed_dim       = cfg['MODEL']['embed_dim'],
            depth           = cfg['MODEL']['depth'],
            n_heads         = cfg['MODEL']['n_heads'],
            mlp_ratio       = cfg['MODEL']['mlp_ratio'],
            qkv_bias        = cfg['MODEL']['qkv_bias'],
            drop_path_rate  = cfg['MODEL']['drop_path_rate'],
            p               = cfg['MODEL']['p'],
            attn_p          = cfg['MODEL']['attn_p'],
            patch_embed_fun = cfg['MODEL']['patch_embed_fun'],
            with_dist_token = cfg['MODEL']['with_dist_token']
        )

        print('DeiT model built.')

    elif cfg['MODEL']['TYPE'] in ['ViT3D_monai']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            in_channels     = cfg['MODEL']['in_chans'],
            img_size        = cfg['MODEL']['img_size'],
            patch_size      = cfg['MODEL']['patch_size'],
            hidden_size     = cfg['MODEL']['embed_dim'],
            mlp_dim         = 4 * cfg['MODEL']['embed_dim'],
            num_layers      = cfg['MODEL']['depth'],
            num_heads       = cfg['MODEL']['n_heads'],
            pos_embed       = 'conv',
            classification  = True,
            num_classes     = cfg['MODEL']['n_classes'],
            dropout_rate    = cfg['MODEL']['p'],
            spatial_dims    = 3,
            post_activation = 'Tanh',
            qkv_bias        = cfg['MODEL']['qkv_bias']
        )

    elif cfg['MODEL']['TYPE'] in ['ResNet152']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + '_' + args.mode + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            spatial_dims     = 3,
            n_input_channels = 1,
            num_classes      = cfg['MODEL']['n_classes']
        )
    
    return model

def make_mae_model(cfg, args):
    """Build a 3D MAE (ViT-B based)
    to be used for pre-training

    """

    if cfg['MODEL']['TYPE'] in ['MaskedAutoencoderViT3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model_mae = _models_factory[cfg['MODEL']['TYPE']](
            img_size          = cfg['MODEL']['img_size'],
            patch_size        = cfg['MODEL']['patch_size'], 
            in_chans          = cfg['MODEL']['in_chans'],
            embed_dim         = cfg['MODEL']['embed_dim'], 
            depth             = cfg['MODEL']['depth'], 
            num_heads         = cfg['MODEL']['n_heads'],
            qkv_bias          = cfg['MODEL']['qkv_bias'],
            drop_path_rate    = cfg['MODEL']['drop_path_rate'],
            decoder_embed_dim = cfg['MODEL']['decoder_embed_dim'], 
            decoder_depth     = cfg['MODEL']['decoder_depth'], 
            decoder_num_heads = cfg['MODEL']['decoder_num_heads'],
            mlp_ratio         = cfg['MODEL']['mlp_ratio'], 
            norm_pix_loss     = cfg['MODEL']['norm_pix_loss'],
            patch_embed_fun   = 'conv3d'
        )

        print('MAE ', cfg['MODEL']['TYPE'], ' model built.')
    
    elif cfg['MODEL']['TYPE'] in ['MaskedAutoencoderDeiT3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model_mae = _models_factory[cfg['MODEL']['TYPE']](
            img_size           = cfg['MODEL']['img_size'],
            patch_size         = cfg['MODEL']['patch_size'], 
            in_chans           = cfg['MODEL']['in_chans'],
            embed_dim          = cfg['MODEL']['embed_dim'], 
            depth              = cfg['MODEL']['depth'], 
            num_heads          = cfg['MODEL']['n_heads'],
            qkv_bias           = cfg['MODEL']['qkv_bias'],
            drop_path_rate     = cfg['MODEL']['drop_path_rate'],
            decoder_embed_dim  = cfg['MODEL']['decoder_embed_dim'], 
            decoder_depth      = cfg['MODEL']['decoder_depth'], 
            decoder_num_heads  = cfg['MODEL']['decoder_num_heads'],
            mlp_ratio          = cfg['MODEL']['mlp_ratio'], 
            norm_pix_loss      = cfg['MODEL']['norm_pix_loss'],
            decoder_dist_token = False,
            patch_embed_fun    = 'conv3d'
        )

        print('DeiT MAE ', cfg['MODEL']['TYPE'], ' model built.')
    
    return model_mae
