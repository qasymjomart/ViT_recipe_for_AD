"""

@qasymjomart

This file imports three MRI datasets:
ADNI1, ADNI2, and OASIS3

The main backbone is borrowed from my previous paper:
https://github.com/qasymjomart/ssl_jigsaw_hipposeg

"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
import os
import glob
import copy
from skimage import io, color, segmentation
import nibabel as nib
from natsort import natsorted

import nibabel as nib

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MaskGenerator3D:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 3
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        
        return mask

class DatasetList():
    """
    DatasetList class that prepares datalists for MONAI Datasets

    Args:
        args (args): args
        cfg (yaml config): cfg
        dataroot : data root
        labelsroot : labels csv root

    Returns:
        dataset: monai.Dataset
    """
    def __init__(self,
                dataset,
                dataroot,
                labelsroot,
                classes_to_use=['CN', 'AD']):
        super(DatasetList, self).__init__()
        
        self.classes_to_use = classes_to_use
        self.labels = pd.read_csv(labelsroot) # Reading the CSV file that joins filenames and labels
        self.labels
        folders_list_ = natsorted(glob.glob(dataroot + '*/hdbet_*.nii.gz'))
        self.folders_list = copy.deepcopy(folders_list_)
        for f_list in folders_list_:
            if 'mask' in f_list:
                self.folders_list.remove(f_list)
        print('Total number of files found: ', len(self.folders_list))

        self.datalist = []

        if dataset == 'ADNI1':
            self.make_ADNI1_datalist()
        elif dataset == 'ADNI2':
            self.make_ADNI2_datalist()
        elif dataset == 'OASIS3':
            self.labels.replace('NORMCOG', 'CN', inplace=True)
            self.make_OASIS3_datalist()
        
        # remove labels not to be used
        for item in self.datalist:
            item["label"] = self.label_to_int(item["label"])
        
        print('Finished making datalist for ',  dataset)
        print('Total dataset size: ', self.__len__())
        print('Label distribution: ', self.__getlabelsratio__(), self.__getlabelsfractions__())

    def make_ADNI1_datalist(self):

        for ii in range(len(self.labels)):
            subj_ii = self.labels.iloc[ii]['Subject']
            image_id_ii = self.labels.iloc[ii]['Image Data ID']

            label_ii = self.labels.Group[(self.labels.Subject == subj_ii) & (self.labels['Image Data ID'] == image_id_ii)].to_list()[0] # str
            path_to_file = [x for x in self.folders_list if subj_ii in x and image_id_ii in x]
            assert len(path_to_file) == 1, "More than one file found for " + subj_ii + " and " + image_id_ii

            self.datalist.append({"image": path_to_file, "label": label_ii})
        
        datalist_ = copy.deepcopy(self.datalist)
        
        # remove labels not to be used
        for item in datalist_:
            if item["label"] in self.classes_to_use:
                pass
            else:
                self.datalist.remove(item)
    
    def make_ADNI2_datalist(self):

        for ii in range(len(self.labels)):
            subj_ii = self.labels.iloc[ii]['Subject']
            image_id_ii = self.labels.iloc[ii]['Image Data ID']

            label_ii = self.labels.Group[(self.labels.Subject == subj_ii) & (self.labels['Image Data ID'] == image_id_ii)].to_list()[0] # str
            path_to_file = [x for x in self.folders_list if subj_ii in x and image_id_ii in x]
            assert len(path_to_file) == 1, "More than one file found for " + subj_ii + " and " + image_id_ii

            self.datalist.append({"image": path_to_file, "label": label_ii})
        
        datalist_ = copy.deepcopy(self.datalist)
        
        # remove labels not to be used
        for item in datalist_:
            if item["label"] in self.classes_to_use:
                pass
            else:
                self.datalist.remove(item)
    
    def make_OASIS3_datalist(self):

        for ii in range(len(self.labels)):
            
            subj_ii = self.labels.iloc[ii]['OASISID']
            # day_id_ii = os.path.split(self.datapath_list[ii])[1].split('ADNI1_')[1][11:18]
            file_name = self.labels[self.labels.OASISID == subj_ii]['filename'].to_list()[0][:-5]
            path_to_file = [x for x in self.folders_list if file_name in x]
            assert file_name in path_to_file[0], subj_ii + " not in " + path_to_file

            label_ii = self.labels[self.labels.OASISID == subj_ii].Label.to_list()[0] # str
            assert len(path_to_file) == 1, "More than one file found for " + subj_ii + " and " + file_name

            self.datalist.append({"image": path_to_file, "label": label_ii})
        
        datalist_ = copy.deepcopy(self.datalist)
        
        # remove labels not to be used
        for item in datalist_:
            if item["label"] in self.classes_to_use:
                pass
            else:
                self.datalist.remove(item)
    
    def __len__(self):
        # get the dataset size
        return len(self.datalist)
    
    def __getlabelsratio__(self):
        ratios = {}
        for label in self.classes_to_use:
            ratios[label] = sum([1 for x in self.datalist if x['label'] == self.label_to_int(label)])
        return ratios
    
    def __getlabelsfractions__(self):
        ratios = {}
        for label in self.classes_to_use:
            ratios[label] = round(sum([1 for x in self.datalist if x['label'] == self.label_to_int(label)]) / self.__len__(), 2)
        return ratios

    def label_to_int(self, label):
        return self.classes_to_use.index(label)
    
    def int_to_label(self, label):
        return self.classes_to_use[label]