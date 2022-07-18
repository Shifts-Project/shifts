"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina, Nataliia Molchanova
"""

import argparse
import os
import torch
from torch import nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import UNet
import numpy as np
import random
from metrics import dice_metric
from data_load import get_train_dataloader, get_val_dataloader

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# trainining
parser.add_argument('--learning_rate', type=float, default=1e-5, 
                    help='Specify the initial learning rate')
parser.add_argument('--n_epochs', type=int, default=300, 
                    help='Specify the number of epochs to train for')
# initialisation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
# data
parser.add_argument('--path_train_data', type=str, required=True, 
                    help='Specify the path to the training data files directory')
parser.add_argument('--path_train_gts', type=str, required=True, 
                    help='Specify the path to the training gts files directory')
parser.add_argument('--path_val_data', type=str, required=True, 
                    help='Specify the path to the validation data files directory')
parser.add_argument('--path_val_gts', type=str, required=True, 
                    help='Specify the path to the validation gts files directory')
parser.add_argument('--num_workers', type=int, default=10, 
                    help='Number of workers')
# logging
parser.add_argument('--path_save', type=str, default='', 
                    help='Specify the path to the trained model will be saved')
parser.add_argument('--val_interval', type=int, default=5, 
                    help='Validation every n-th epochs')
parser.add_argument('--threshold', type=float, default=0.4, 
                    help='Probability threshold')


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def main(args):
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    path_save = args.path_save
    
    '''' Initialise dataloaders '''
    train_loader = get_train_dataloader(flair_path=args.path_train_data, 
                                        gts_path=args.path_train_gts, 
                                        num_workers=args.num_workers)
    val_loader = get_val_dataloader(flair_path=args.path_val_data, 
                                    gts_path=args.path_val_gts, 
                                    num_workers=args.num_workers)
  
    ''' Initialise the model '''
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=0).to(device)
    
    loss_function = DiceLoss(to_onehot_y=True, 
                             softmax=True, sigmoid=False,
                             include_background=False)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    act = nn.Softmax(dim=1)
    
    epoch_num = args.n_epochs
    val_interval = args.val_interval
    thresh = args.threshold
    gamma_focal = 2.0
    dice_weight = 0.5
    focal_weight = 1.0
    roi_size = (96, 96, 96)
    sw_batch_size = 4
    
    
    best_metric, best_metric_epoch = -1, -1
    epoch_loss_values, metric_values = [], []

    ''' Tranining loop '''
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            n_samples = batch_data["image"].size(0)
            for m in range(0,batch_data["image"].size(0), 2):
                step += 2
                inputs, labels = (
                    batch_data["image"][m:(m+2)].to(device),
                    batch_data["label"][m:(m+2)].type(torch.LongTensor).to(device))
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Dice loss
                loss1 = loss_function(outputs, labels)
                # Focal loss
                ce_loss = nn.CrossEntropyLoss(reduction='none')
                ce = ce_loss(outputs, torch.squeeze(labels, dim=1))
                pt = torch.exp(-ce)
                loss2 = (1 - pt)**gamma_focal * ce 
                loss2 = torch.mean(loss2)
                loss = dice_weight * loss1 + focal_weight * loss2              
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                if step % 100 == 0:
                    step_print = int(step/2)
                    print(f"{step_print}/{(len(train_loader)*n_samples) // (train_loader.batch_size*2)}, train_loss: {loss.item():.4f}")

        epoch_loss /= step_print
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        ''' Validation '''
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device)
                        )
                    
                    val_outputs = sliding_window_inference(val_inputs, roi_size, 
                                                           sw_batch_size, 
                                                           model, mode='gaussian')
                   
                    gt = np.squeeze(val_labels.cpu().numpy())
                    
                    seg = act(val_outputs).cpu().numpy()
                    seg= np.squeeze(seg[0,1])
                    seg[seg >= thresh] = 1
                    seg[seg < thresh] = 0
                    
                    value = dice_metric(ground_truth=gt.flatten(), predictions=seg.flatten())

                    metric_count += 1
                    metric_sum += value.sum().item()
                
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(path_save, "Best_model_finetuning.pth"))
                    print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                                    )
 
          
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)