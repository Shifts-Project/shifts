"""
Contains implementations of transforms and dataloaders needed for training, validation and inference.
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
from scipy import ndimage


def get_train_transforms():
    """ Get transforms for training on FLAIR images and ground truth:
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Normalises intensity
    - Applies augmentations
    - Crops out 32 patches of shape [96, 96, 96] that contain lesions
    - Converts to torch.Tensor()
    """
    return Compose(
    [
        LoadImaged(keys=["image","label"]),
        AddChanneld(keys=["image","label"]),
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
        RandFlipd(keys=["image", "label"],prob=0.5,spatial_axis=(0,1,2)),
        RandRotate90d(keys=["image", "label"],prob=0.5,spatial_axes=(0,1)),
        RandRotate90d(keys=["image", "label"],prob=0.5,spatial_axes=(1,2)),
        RandRotate90d(keys=["image", "label"],prob=0.5,spatial_axes=(0,2)),
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), 
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(np.pi/12, np.pi/12, np.pi/12), 
                    scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
        ToTensord(keys=["image", "label"]),
    ]
    )

def get_val_transforms():
    """ Get transforms for testing on FLAIR images and ground truth:
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Converts to torch.Tensor()
    """
    return Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image","label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image", "label"]),
    ]
    )

def get_flair_transforms():
    """ Get transforms for testing on FLAIR images:
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Converts to torch.Tensor()
    """
    return Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ToTensord(keys=["image"]),
    ]
    )

def get_train_dataloader(flair_path, gts_path, num_workers, cache_rate=0.1):
    """
    Get dataloader for training 
    Args:
      flair_path: `str`, path to directory with FLAIR images from Train set.
      gts_path:  `str`, path to directory with ground truth lesion segmentation 
                    binary masks images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
    Returns:
      monai.data.DataLoader() class object.
    """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(gts_path, "*gt_isovox.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i))) # Collect all corresponding ground truths
    
    files = [{"image": fl,"label": seg} for fl, seg in zip(flair, segs)]
    
    print("Number of training files:", len(files))
    
    ds = CacheDataset(data=files, transform=get_train_transforms(), 
                            cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=True, 
                              num_workers=num_workers)

def get_val_dataloader(flair_path, gts_path, num_workers, cache_rate=0.1):
    """
    Get dataloader for validation and testing

    Args:
      flair_path: `str`, path to directory with FLAIR images.
      gts_path:  `str`, path to directory with ground truth lesion segmentation 
                    binary masks images.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
    Returns:
      monai.data.DataLoader() class object.
    """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    segs = sorted(glob(os.path.join(gts_path, "*gt_isovox.nii.gz")),
                  key=lambda i: int(re.sub('\D', '', i))) # Collect all corresponding ground truths
    
    files = [{"image": fl,"label": seg} for fl, seg in zip(flair, segs)]
    
    print("Number of validation files:", len(files))
    
    ds = CacheDataset(data=files, transform=get_val_transforms(), 
                            cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False, 
                              num_workers=num_workers)

def get_flair_dataloader(flair_path, num_workers, cache_rate=0.1):
    """
    Get dataloader with FLAIR images only for inference
    
    Args:
      flair_path: `str`, path to directory with FLAIR images from Train set.
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
    Returns:
      monai.data.DataLoader() class object.
    """
    flair = sorted(glob(os.path.join(flair_path, "*FLAIR_isovox.nii.gz")),
                 key=lambda i: int(re.sub('\D', '', i)))  # Collect all flair images sorted
    
    files = [{"image": fl} for fl in flair]
    
    print("Number of FLAIR files:", len(files))
    
    ds = CacheDataset(data=files, transform=get_flair_transforms(), 
                            cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=False, 
                              num_workers=num_workers)

def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a 
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = ndimage.label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = ndimage.labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2