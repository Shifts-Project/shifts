"""
Perform inference for an ensemble of baseline models and save 3D Nifti images of
predicted probability maps averaged across ensemble models (saved to "*pred_prob.nii.gz" files),
binary segmentation maps predicted obtained by thresholding of average predictions and 
removing all connected components smaller than 9 voxels (saved to "pred_seg.nii.gz"),
uncertainty maps for reversed mutual information measure (saved to "uncs_rmi.nii.gz").
"""

import argparse
import os
import re
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.data import write_nifti
import numpy as np
from data_load import remove_connected_components, get_flair_dataloader
from uncertainty import ensemble_uncertainties_classification

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# save options
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory to store predictions')
# model
parser.add_argument('--num_models', type=int, default=3,
                    help='Number of models in ensemble')
parser.add_argument('--path_model', type=str, default='',
                    help='Specify the dir to al the trained models')
# data
parser.add_argument('--path_data', type=str, required=True,
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_bm', type=str, required=True,
                    help='Specify the path to the directory with brain masks')
# parallel computation
parser.add_argument('--num_workers', type=int, default=10,
                    help='Number of workers to preprocess images')
# hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    os.makedirs(args.path_pred, exist_ok=True)
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    '''' Initialise dataloaders '''
    val_loader = get_flair_dataloader(flair_path=args.path_data,
                                      num_workers=args.num_workers,
                                      bm_path=args.path_bm)

    ''' Load trained models  '''
    K = args.num_models
    models = []
    for i in range(K):
        models.append(UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=0).to(device)
                      )

    for i, model in enumerate(models):
        model.load_state_dict(torch.load(os.path.join(args.path_model,
                                                      f"seed{i + 1}",
                                                      "Best_model_finetuning.pth")))
        model.eval()

    act = torch.nn.Softmax(dim=1)
    th = args.threshold
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    ''' Predictions loop '''
    with torch.no_grad():
        for count, batch_data in enumerate(val_loader):
            inputs = batch_data["image"].to(device)
            foreground_mask = batch_data["brain_mask"].numpy()[0, 0]

            # get ensemble predictions
            all_outputs = []
            for model in models:
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
                outputs = act(outputs).cpu().numpy()
                outputs = np.squeeze(outputs[0, 1])
                all_outputs.append(outputs)
            all_outputs = np.asarray(all_outputs)

            # get image metadata
            original_affine = batch_data['image_meta_dict']['original_affine'][0]
            affine = batch_data['image_meta_dict']['affine'][0]
            spatial_shape = batch_data['image_meta_dict']['spatial_shape'][0]
            filename_or_obj = batch_data['image_meta_dict']['filename_or_obj'][0]
            filename_or_obj = os.path.basename(filename_or_obj)

            # obtain and save probability maps averaged across models in an ensemble
            outputs_mean = np.mean(all_outputs, axis=0)

            filename = re.sub("FLAIR_isovox.nii.gz", 'pred_prob.nii.gz',
                              filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(outputs_mean, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)

            # obtain and save binary segmentation masks
            seg = outputs_mean.copy()
            seg[seg >= th] = 1
            seg[seg < th] = 0
            seg = np.squeeze(seg)
            seg = remove_connected_components(seg)

            filename = re.sub("FLAIR_isovox.nii.gz", 'pred_seg.nii.gz',
                              filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(seg, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        mode='nearest',
                        output_spatial_shape=spatial_shape)

            # obtain and save uncertainty map (voxel-wise reverse mutual information)
            uncs_map = ensemble_uncertainties_classification(np.concatenate(
                (np.expand_dims(all_outputs, axis=-1),
                 np.expand_dims(1. - all_outputs, axis=-1)),
                axis=-1))['reverse_mutual_information']

            filename = re.sub("FLAIR_isovox.nii.gz", 'uncs_rmi.nii.gz',
                              filename_or_obj)
            filepath = os.path.join(args.path_pred, filename)
            write_nifti(uncs_map * foreground_mask, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
