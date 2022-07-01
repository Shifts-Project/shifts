"""
Computation of performance metrics (nDSC, lesion F1 score) for an ensemble of models.
Metrics are displayed in console.
"""

import argparse
import os
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import numpy as np
import random
from data_load import remove_connected_components, get_val_dataloader
from metrics import dice_norm_metric, lesion_f1_score

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# model
parser.add_argument('--num_models', type=int, default=5, 
					help='Number of models in ensemble')
parser.add_argument('--path_model', type=str, default='', 
					help='Specify the dir to al the trained models')
# data
parser.add_argument('--path_data', type=str, required=True, 
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_gts', type=str, required=True, 
                    help='Specify the path to the directory with ground truth binary masks')
# parallel computation
parser.add_argument('--num_workers', type=int, default=10, 
                    help='Number of workers to preprocess images')
parser.add_argument('--n_jobs', type=int, default=1, 
					help='Number of parallel workers for F1 score computation')
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
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    '''' Initialise dataloaders '''
    val_loader = get_val_dataloader(flair_path=args.path_data, 
                                    gts_path=args.path_gts, 
                                    num_workers=args.num_workers)
  
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
        model.load_state_dict(torch.load(os.path.join(root_dir, f"seed{i+1}", "Best_model_finetuning.pth")))
        model.eval()

    act = nn.Softmax(dim=1)
    th = args.threshold
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    ndsc, f1 = [], []

    ''' Evaluatioin loop '''
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
	    with torch.no_grad():
	        for count, batch_data in enumerate(val_loader):
	            inputs, gt  = (
	                    batch_data["image"].to(device),
	                    batch_data["label"].cpu().numpy()
	                    )
	            # get ensemble predictions
	            all_outputs = []
	            for model in models:
	                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
	                outputs = act(outputs).cpu().numpy()
	                outputs = np.squeeze(outputs[0,1])
	                all_outputs.append(outputs)
	            all_outputs = np.asarray(all_outputs)

	            # obtain binary segmentation mask
	            seg = np.mean(all_outputs, axis=0)
	            seg[seg>th]=1
	            seg[seg<th]=0
	            seg= np.squeeze(seg)
	            seg = remove_connected_components(seg)
	  
	            gt = np.squeeze(gt)

	            # compute metrics
	            ndsc += [dice_norm_metric(ground_truth=gt, predictions=seg)]
	            f1 += [lesion_f1_score(ground_truth=gt, 
						            	predictions=seg, 
						            	IoU_threshold=0.5, 
						            	parallel_backend=parallel_backend)]

	ndsc = np.asarray(ndsc) * 100.
	f1 = np.asarray(f1) * 100.

    print(f"nDSC:\t{np.mean(ndsc):.4f} +- {np.std(ndsc):.4f}")
    print(f"Lesion F1 score:\t{np.mean(f1):.4f} +- {np.std(f1):.4f}")
 
          
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)