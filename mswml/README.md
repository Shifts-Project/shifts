# Multiple Sclerosis White Matter Lesions Segmentation

*#TODO:* general description of the task, link to the GC, link to the paper
 
## Files description

`metrics.py` contains implementations of metrics used for validation during training and evaluation: Dice score, Normalised Dice score, Lesion F1 score.

`uncertainty.py` contains implementations of uncertainty measures: mutual information (MI), expected pair-wise KL divergence (EPKL) and reverse mutual information (RMI) for knowledge uncertainty; expected entropy (ExE) for data uncertainty; entropy of expected (EoE) and negated confidence (NC) for total uncertainty.

`data_load.py` contains implementations of transforms and dataloaders needed for training, validation and inference.

`train.py`, `test.py`, `inference.py` are programs used for reproduction of a baseline model.

## Baseline model

1. Training baseline ensemble of 5 UNET models.

```bash
#!/bin/bash
for seed in 1 2 3 4 5
do
	python mswml/train.py \
	--seed $seed \
	--path_train_data /path/to/train/FLAIR \
	--path_train_gts /path/to/train/ground/truth/masks \
	--path_val_data /path/to/val/FLAIR \
	--path_val_gts /path/to/val/ground/truth/masks \
	--path_save "/path/to/models/save/dir/${seed}"
done
```
2. Performence metrics computation for a trained ensemble

```bash
python mswml/test.py \
--path_model /path/to/models/save/dir/ \
--path_data /path/to/test/FLAIR \
--path_gts /path/to/test/ground/truth/masks \
--threshold 0.35
```

Additional parameters like `--num_workers` and `--n_jobs` control the number of workers used for parallel processing of images and paralle computaiton of lesion F1 score respectively.

Probablity threshold `threshold` is used for obtaining binary lesion masks from probability output.

3. Saving predicted probability and binary segmentation maps

```bash
python mswml/inference.py \
--path_pred /path/to/dir/to/save/predictions/ \
--path_model /path/to/models/save/dir/ \
--path_data /path/to/test/FLAIR \
--num_workers 10\
--threshold 0.35
```

## Baline model performance

#TODO: report results with the new version of MONAI.

## Requirements

#TODO verify completness

See `requirements.txt` file.

## Data visualisation

Both 3D FLAIR images and ground truth segmentation masks are distributed in [Nifti](https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_roi.html) format.

For the visualisation of images, we suggest either using [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) software for medical images visualisation or [nilearn](https://nilearn.github.io/stable/index.html) Python library for displaying cuts of 3D images. For the last option consider using `nilearn.plotting.plot_img()` function to display slices of a 3D image and `nilearn.plotting_plot_roi()` for displaing slices of a 3D image with additional overlays, e.g. ground truth masks or predicted binary masks.

