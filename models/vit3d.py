"""

3D ViT transformer that inputs 5D (n_batches, n_channels, height, weight, depth)

Based primarily on a video tutorial from Vision Transformer

and 

Official code PyTorch implementation from CDTrans paper:
https://github.com/CDTrans/CDTrans

"""

import math
import copy
from functools import partial
from itertools import repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

# from .resnetv2 import ResNetV2
from .convnets import UNet3DEncoder
from .pe import PatchEmbed, PatchEmbed3X, ProgressivePatchEmbed, ProgressivePatchEmbed3D

from utils.weight_init import trunc_normal_, init_weights_vit_timm, get_init_weights_vit, named_apply
from utils.utils import get_3d_sincos_pos_embed

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed3D(nn.Module):
    """
    Split image into 3D patches and then embed them.

    Parameters
    ----------
    img_size : int (square)
    patch_size : int (square)
    in_chans : int
    embed_dim : int

    Atttributes:
    -----------
    n_patches : int
    proj : nn.Conv2d

    """
    def __init__(self, img_size, patch_size, embed_dim=768, patch_embed_fun='conv3d'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)

        # sample random tensor to calculate the output shape
        sample_torch = torch.rand((1, 1, *self.img_size)) # --> e.g. (1,1,128,128,128)

        if patch_embed_fun == 'conv3d':
            self.proj = nn.Conv3d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        elif patch_embed_fun == 'unet3d':
            self.proj = UNet3DEncoder(
                in_chans=[[1, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, embed_dim]],
                num_blocks=4,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False
            )
        elif patch_embed_fun == 'mype3d':
            self.proj = nn.Sequential(
                nn.Conv3d(1, embed_dim//4, 5),
                nn.BatchNorm3d(embed_dim//4),
                nn.ReLU(),
                nn.AvgPool3d(3),
                nn.Conv3d(embed_dim//4, embed_dim//2, 5),
                nn.BatchNorm3d(embed_dim//2),
                nn.ReLU(),
                nn.AvgPool3d(3),
                nn.Conv3d(embed_dim//2, embed_dim, 5),
                nn.BatchNorm3d(embed_dim),
                nn.ReLU()
            )
        
        out = self.proj(sample_torch)
        self.n_patches = out.flatten(2).shape[2]

    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_patches, embed_dims)
        """
        x = self.proj(x) # out: (n_samples, embed_dim, n_patches[0], n_patches[1], n_patches[2])
        # x = x.view(-1, self.e)
        x = x.flatten(2) # out: (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # out: (n_samples, n_patches, embed_dim)///

        return x

class Attention(nn.Module):
    """
    Attention mechanism

    Parameters
    -----------
    dim : int (dim per token features)
    n_heads : int
    qkv_bias : bool
    attn_p : float (Dropout applied to q, k, v)
    proj_p : float (Dropout applied to output tensor)

    Attributes
    ----------
    scale : float
    qkv : nn.Linear
    proj : nn.Linear
    attn_drop, proj_drop : nn.Dropout
    
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        """
        Input
        ------
        x : Shape (n_samples, n_patches + 1, dim)

        Returns:
        -------
        Shape (n_samples, n_patches + 1, dim)

        """
        n_samples, n_tokens, dim =  x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)

        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # each with (n_samples, n_heads, n_patches + 1, head_dim)

        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
            q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        ) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron

    Parameters
    ----------
    in_features : int
    hidden_features : int
    out_features : int
    p : float

    Attributes
    ---------
    fc1 : nn.Linear
    act : nn.GELU
    fc2 : nn.Linear
    drop : nn.Dropout
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, in_features)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, out_features)
        """
        x = self.fc1(
            x
            ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, out_features)
        x = self.drop(x) # (n_samples, n_patches + 1, out_features)

        return x

class Block(nn.Module):
    """
    Transformer block

    Parameters
    ----------
    dim : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes
    ----------
    norm1, norm2 : LayerNorm
    attn : Attention
    mlp : MLP
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0., p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Input
        ------
        Shape (n_samples, n_patches + 1, dim)

        Returns:
        ---------
        Shape (n_samples, n_patches + 1, dim)
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Vision_Transformer3D(nn.Module):
    """
    3D Vision Transformer

    Parameters
    -----------
    img_size : int
    patch_size : int
    in_chans : int
    n_classes : int
    embed_dim : int
    depth : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes:
    -----------
    patch_embed : PatchEmbed
    cls_token : nn.Parameter
    pos_emb : nn.Parameter
    pos_drop : nn.Dropout
    blocks : nn.ModuleList
    norm : nn.LayerNorm
    """
    def __init__(self, 
                img_size=384, 
                patch_size=16, 
                in_chans=3, 
                n_classes=1000, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                mlp_ratio=4., 
                qkv_bias=True, 
                drop_path_rate=0.,
                p=0., 
                attn_p=0.,
                patch_embed_fun='conv3d',
                weight_init='',
                global_avg_pool=False,
                pos_embed_type='learnable',
                use_separation=True
                ):
        super().__init__()

        if patch_embed_fun in ['conv3d', 'unet3d', 'mype3d']:
            self.patch_embed = PatchEmbed3D(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun
            )
        elif patch_embed_fun in ['conv2d', 'resnet502d', 'unet2d', 'mype2d']:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun,
                use_separation=use_separation
            )
        elif patch_embed_fun in ['3xconv2d', '3xmype2d']:
            self.patch_embed = PatchEmbed3X(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun,
                use_separation=use_separation
            )
        elif patch_embed_fun in ['progressive_conv2d']:
            self.patch_embed = ProgressivePatchEmbed(
                img_size=img_size,
                patch_size=16,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun
            )
        elif patch_embed_fun in ['progressive_conv3d']:
            self.patch_embed = ProgressivePatchEmbed3D(
                img_size=img_size,
                patch_size=16,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun
            )

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim)) if global_avg_pool == False else None
        embed_len = self.patch_embed.n_patches if global_avg_pool else 1 + self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(
                torch.rand(1, embed_len, embed_dim), requires_grad=True
            )
        
        if pos_embed_type == 'abs':
            self.pos_embed = nn.Parameter(
                torch.rand(1, embed_len, embed_dim), requires_grad=False
            )
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(np.cbrt(self.patch_embed.n_patches)), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            print('Abs pos embed built.')
            
        self.pos_drop = nn.Dropout(p=p)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=self.dpr[ii],
                    p=p,
                    attn_p=attn_p
                )
                for ii in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.pos_embed, std=.02)

        # self.apply(self._init_weights_vit_timm)

        self.init_weights(weight_init)
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)
        print("Model weights initialized")

    def _init_weights_vit_timm(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def forward(self, x):
        """
        Input
        -----
        Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_classes)
        
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(
                n_samples, -1, -1
            ) # (n_samples, 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        # just the CLS token
        cls_token_final = x[:, 0] if self.cls_token is not None else x.mean(dim=1)
        # cls_token_final = self.bottleneck(cls_token_final)
        x = self.head(cls_token_final)

        return x
    
    def save(self, optimizer, scaler, checkpoint):
        state = {"net": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()}
        torch.save(state, checkpoint)

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class DeiT_Transformer3D(nn.Module):
    """
    3D Vision DeiT Transformer

    Parameters
    -----------
    img_size : int
    patch_size : int
    in_chans : int
    n_classes : int
    embed_dim : int
    depth : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : book
    p, attn_p : float

    Attributes:
    -----------
    patch_embed : PatchEmbed
    cls_token : nn.Parameter
    pos_emb : nn.Parameter
    pos_drop : nn.Dropout
    blocks : nn.ModuleList
    norm : nn.LayerNorm
    """
    def __init__(self, 
                img_size=384, 
                patch_size=16, 
                n_classes=1000, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                mlp_ratio=4., 
                qkv_bias=True, 
                drop_path_rate=0.,
                p=0., 
                attn_p=0.,
                patch_embed_fun='conv3d',
                weight_init='',
                training=True,
                with_dist_token=True
                ):
        super().__init__()

        if patch_embed_fun in ['conv3d', 'unet3d', 'mype3d']:
            self.patch_embed = PatchEmbed3D(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                patch_embed_fun=patch_embed_fun
            )
        
        self.training = training
        self.with_dist_token = with_dist_token

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        if with_dist_token:
            self.dist_token = nn.Parameter(torch.rand(1, 1, embed_dim))  # new distillation token
            embed_len = self.patch_embed.n_patches + 2
            print("With distillation token")
        else:
            self.dist_token = None
            embed_len = self.patch_embed.n_patches + 1
            print("No distillation token")
            
        self.pos_embed = nn.Parameter(
                torch.rand(1, embed_len, embed_dim), requires_grad=True
            )
            
        self.pos_drop = nn.Dropout(p=p)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=self.dpr[ii],
                    p=p,
                    attn_p=attn_p
                )
                for ii in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
        self.head_dist = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()
        
        trunc_normal_(self.cls_token, std=.02)
        
        if with_dist_token:
            trunc_normal_(self.dist_token, std=.02)
        # trunc_normal_(self.pos_embed, std=.02)
        # self.apply(self._init_weights_vit_timm)
        
        self.init_weights(weight_init)
        self.head_dist.apply(self._init_weights)
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.dist_token is not None:
            nn.init.normal_(self.dist_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)
        print("Model weights initialized")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_vit_timm(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def forward(self, x):
        """
        Input
        -----
        Shape (n_samples, in_chans, img_size, img_size)

        Returns:
        --------
        Shape (n_samples, n_classes)
        
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        if self.with_dist_token:
            dist_token = self.dist_token.expand(n_samples, -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1) # (n_samples, 2 + n_patches, embed_dim)
        else:
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        # just the CLS token
        cls_token_final = x[:, 0]
        
        if self.with_dist_token:    
            dist_token_final = x[:, 1]
            x_dist = self.head_dist(dist_token_final)
        else:
            x_dist = self.head_dist(cls_token_final)
            
        x = self.head(cls_token_final)
        
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2
    
    def save(self, optimizer, scaler, checkpoint):
        state = {"net": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()}
        torch.save(state, checkpoint)