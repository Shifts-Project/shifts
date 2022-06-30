#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 07:40:46 2022

@author: Nataliia Molchanova
"""
import numpy as np
import os
from glob import glob 
import re
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChanneld,Compose,LoadImaged,RandCropByPosNegLabeld,
    Spacingd,ToTensord,NormalizeIntensityd, RandFlipd,
    RandRotate90d,RandShiftIntensityd,RandAffined,RandSpatialCropd, 
    RandScaleIntensityd)

def get_train_transforms():
    return Compose(
    [
        LoadImaged(keys=["image","label"]),AddChanneld(keys=["image","label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        RandShiftIntensityd(keys="image",offsets=0.1,prob=1.0),
        RandScaleIntensityd(keys="image",factors=0.1,prob=1.0),
        RandCropByPosNegLabeld(keys=["image", "label"],
                               label_key="label", image_key="image",
                               spatial_size=(128, 128, 128), num_samples=32,
                               pos=4,neg=1),
        RandSpatialCropd(keys=["image", "label"], 
                         roi_size=(96,96,96), 
                         random_center=True, random_size=False),
        RandFlipd (keys=["image", "label"],prob=0.5,spatial_axis=(0,1,2)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(0,1)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(1,2)),
        RandRotate90d (keys=["image", "label"],prob=0.5,spatial_axes=(0,2)),
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), 
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(np.pi/12, np.pi/12, np.pi/12), 
                    scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
        ToTensord(keys=["image", "label"]),
    ]
    )

def get_val_transforms():
    return Compose(
    [
        LoadImaged(keys=["image", "label"]),AddChanneld(keys=["image","label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label"]),
    ]
    )

def get_train_dataloader(flair_path, gts_path, num_workers, cache_rate=0.1):
    """Get torch.data.DataLoader for training """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(gts_path, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i))) # Collect all corresponding ground truths
    
    files = [{"image": fl,"label": seg} for fl, seg in zip(flair, segs)]
    
    print("Number of training files:", len(files))
    
    ds = CacheDataset(data=files, transform=get_train_transforms(), 
                            cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=True, 
                              num_workers=num_workers)

def get_val_dataloader(flair_path, gts_path, num_workers, cache_rate=0.1):
    """Get torch.data.DataLoader for validation and testing """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(gts_path, "*gt.nii")),
                  key=lambda i: int(re.sub('\D', '', i))) # Collect all corresponding ground truths
    
    files = [{"image": fl,"label": seg} for fl, seg in zip(flair, segs)]
    
    print("Number of validation files:", len(files))
    
    ds = CacheDataset(data=files, transform=get_val_transforms(), 
                            cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False, 
                              num_workers=num_workers)