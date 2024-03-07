# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:17:00 2021

@author: qasymjomart

Based on https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
"""

from __future__ import print_function, division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

class PrototypePseudoLabeling(nn.Module):
    def __init__(self, gamma=0.1, gpu=True, num_classes=3):
        super(PrototypePseudoLabeling, self).__init__()
        self.gamma = gamma
        self.gpu = gpu
        self.prototypes = None
        self.cos_sim = nn.CosineSimilarity()
        self.classes = torch.tensor([x for x in range(num_classes)])
        if self.gpu:
            self.cos_sim.cuda()
    
    def calculate_class_prototypes(self, f, y):
        """
        Calculates class prototypes given samples of Source features f and labels y
        Then update the c_k as c_k = gamma * c_k + (1-gamma) * c_k_new
        """
        prototypes = torch.zeros((len(torch.unique(self.classes)), *f.shape[1:]))

        if self.gpu:
            prototypes = prototypes.cuda()
        
        for c in self.classes: # classes are e.g. 0, 1, 2
            if sum(y==c) > 0:
                class_samples = f[y == c]
                prototypes[c] = torch.mean(class_samples, axis=0)

        self.prototypes = self.gamma * self.prototypes + (1 - self.gamma) * prototypes
    
    def calculate_pseudo_labels(self, ft):
        """Calculate the pseudo labels based on class prototypes
        and target data extracted features

        Parameters
        ----------
        ft : target data extracted features, Shape: (n_samples, n_patches * embed_sim)
        
        Returns:
        ----------
        yt_ps : pseudo labels, Shape: (10)
        """
        yt_ps = []
        cos_distances = torch.zeros((len(ft), len(self.prototypes)))
        if self.gpu:
            cos_distances = cos_distances.cuda()

        for c in self.classes:
            cos_distances[:, c] = self.cos_sim(ft.flatten(1), self.prototypes[c].flatten())
        
        return torch.argmin(cos_distances, dim=1)
    
    def forward(self, fs, ys, ft):
        """
        Do Prototype MMD loss forward

        Args:
            fs (torch.Tensor): Source data batch features
            ys (torch.Tensor)): Source data batch labels
            ft (torch.Tensor)): Target data batch features

        Returns:
            loss: _description_
        """

        assert fs.shape[0] == ft.shape[0], "Batch sizes of source and target features are not the same for Prototype Loss"

        if self.prototypes is None:
            self.prototypes = torch.zeros((len(torch.unique(self.classes)), *fs.shape[1:]))
            if self.gpu:
                self.prototypes = self.prototypes.cuda()

        self.calculate_class_prototypes(fs, ys)
        
        yt_ps = self.calculate_pseudo_labels(ft)
        assert len(yt_ps) == ft.shape[0], "Dimension mismatch between pseudo labels and target data features batch"

        if self.gpu:
            yt_ps = yt_ps.cuda()

        return yt_ps
