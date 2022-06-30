# Multiple Sclerosis White Matter Lesions Segmentation

*#TODO:* general description of the task, link to the GC, link to the paper
 
## Files description

1. Training baseline UNET model.

```bash
python mswml/train.py \
--seed [random seed] \
--path_train_data [path to FLAIR images from train set] \
--path_train_gts [path to ground truth masks from train set] \
--path_val_data [path to FLAIR images from val set] \
--path_val_gts [path to ground truth masks from val set] \
--path_save [path to the folder where results will be stored]
```