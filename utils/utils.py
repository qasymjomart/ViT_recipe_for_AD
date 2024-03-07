# -*- coding: utf-8 -*-
"""
"""
import os
import glob
import numpy as np

import torch
from sklearn.metrics import roc_auc_score
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

import timm

def adjust_learning_rate(optimizer, epoch, init_lr=0.1, step=30, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay ** (epoch // step))
    print('Learning Rate %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_alpha(epoch, alpha=0.01, step = 20, growth_rate=1.2, max_value=0.4):
    """Adjust alpha value of the distillation loss

    Parameters
    ----------
    epoch : int
        Current epoch
    alpha : float
        Current alpha value, if not given then initial alpha is by default 0.01
    step : int
        step after which the alpha is incrased by 1.1
    growth_rate : int
        Rate by which to increase alpha
    max_value : float, optional
        Maximum value of alpha, by default 0.2
    """
    alpha = min(max_value, alpha * (growth_rate ** (epoch // step)))

    return alpha

def compute_auc(output, target):
    """Compute AUC

    Args:
        output (numpy array): Shape (n_samples, n_classes)
        target (numpy array): Shape (n_samples,)
    """
    return roc_auc_score(target, output)

def compute_accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    k = 1
    batch_size = target.size(0)

    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        val_loss = val_loss / 100
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss - self.best_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif val_loss - self.best_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def save_model(args, cfg, model, filename, epoch, steps):
    flist = glob.glob(filename+ '*')
    for f in flist:
        os.remove(f)
    filename = filename + '_%03i_%06d.pth.tar'%(epoch, steps)
    if len([x for x in args.devices.split(",")]) > 1:
        state = {"net": model.module.state_dict()}
    else:
        state = {"net": model.state_dict()}
    torch.save(state, filename)

def load_checkpoint(args, cfg, model, filename):
    files = [f for f in os.listdir(cfg['TRAINING']['CHECKPOINT']) if filename in f]
    if len(files)>0:
        files.sort()
        ckp = files[-1]
        model.load_state_dict(torch.load(cfg['TRAINING']['CHECKPOINT']+ckp)['net'])
        print(ckp, ' found and loaded.')
    
    return model

# Helpers
def get_n_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)

def copy_imagenet_weights(model):
    model_name = "vit_base_patch16_384"
    model_official = timm.create_model(model_name, pretrained=True)

    for (n_0, p_0), (n_c, p_c) in zip(model_official.named_parameters(), model.named_parameters()):
        print(f"{n_0} | {n_c}")
        import pdb; pdb.set_trace()
        assert p_0.numel() == p_c.numel()

        p_c.data[:] = p_0.data

        assert_tensors_equal(p_c.data, p_0.data)

    inp = torch.rand(1, 3, 384, 384)
    res_c = model(inp)
    res_0 = model_official(inp)

    # Asserts
    assert get_n_parameters(model) == get_n_parameters(model_official)
    assert_tensors_equal(res_c, res_0)

    print('Copied ImageNet weights to the model')

    return model

def set_requires_grad(model, requires_grad=True):
    """
    Set requires grad True

    From https://github.com/jvanvugt/pytorch-domain-adaptation/blob/be63aadc18821d6b19c75df51f264ff08370a765/utils.py#L13

    Args:
        model (torch.nn.Module): model
        requires_grad (bool, optional): where to set gradients True. Defaults to True.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

def loop_iterable(iterable):
    """
    Yeild the next iteration batch of dataloader

    From https://github.com/jvanvugt/pytorch-domain-adaptation/blob/be63aadc18821d6b19c75df51f264ff08370a765/utils.py#L13

    Args:
        iterable (torch.utils.data.data.Dataloader): Dataloader to yield the batch from

    Yields:
        batch: batch to yield
    """
    while True:
        yield from iterable

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, dist_token=False):
    """
    from https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_d = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if dist_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), np.zeros([1, embed_dim]), pos_embed], axis=0)
    elif cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W, D/2)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

import math

def adjust_learning_rate_halfcosine(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg['SOLVER']['warmup_epochs']:
        lr = cfg['SOLVER']['lr'] * epoch / cfg['SOLVER']['warmup_epochs'] 
    else:
        lr = cfg['SOLVER']['min_lr'] + (cfg['SOLVER']['lr'] - cfg['SOLVER']['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg['SOLVER']['warmup_epochs']) / (cfg['TRAINING']['EPOCHS'] - cfg['SOLVER']['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return

def load_pretrained_checkpoint(model, pre_trained_model_path, checkpoint_type=None):
    """Loading (transferring) pre-trained MAE model weights

    Parameters
    ----------
    model : torch.nn.Module
        model to finetune
    pre_trained_model_path : str
        path to the pre-trained models checkpoint
    """
    if pre_trained_model_path == 'imagenet_weights/':
        keys_to_remove = ['head.weight', 'head.bias', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
        # keys_to_remove = ['head.weight', 'head.bias', 'pos_embed']
        # checkpoint_model = timm.create_model('vit_large_patch16_224').state_dict()
        checkpoint_model = timm.create_model('vit_base_patch16_224').state_dict()
        # checkpoint_model = timm.create_model('vit_small_patch16_224').state_dict()
        print('Loaded ImageNet pre-trained checkpoint')
        
    else:
        checkpoint = torch.load(pre_trained_model_path, map_location='cpu')
        print("Loaded pre-trained checkpoint from: %s" % pre_trained_model_path)
        checkpoint_model = checkpoint['net']
        keys_to_remove = ['head.weight', 'head.bias', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']

    state_dict = model.state_dict()
        
    for k in keys_to_remove:
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    msg = model.load_state_dict(checkpoint_model, strict=False)

    print(msg.missing_keys)
    
    # if checkpoint_type != 'no_pos_embed':
    #     assert set(msg.missing_keys) == set(keys_to_remove), print(msg.missing_keys)

    return model