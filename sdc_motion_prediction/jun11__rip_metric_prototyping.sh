# We have 3 of Neil's DIM models trained on a 10k scene subset of the training data
# Have just altered RIP to use aggregation at two levels:
# Over the K model scores for each plan
# Over the aggregated per-plan scores, to obtain a scene-level uncertainty

# RIP: get retention metrics

srun --cpus-per-task 10 --gres=gpu:1 --nodelist oat4 --pty bash

# Evaluate all metrics over multiple seeds

# RIP LQ LQ
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 1 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm LQ --rip_per_scene_algorithm LQ --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 2 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm LQ --rip_per_scene_algorithm LQ --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 3 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm LQ --rip_per_scene_algorithm LQ --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping

# RIP UQ UQ
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 1 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm UQ --rip_per_scene_algorithm UQ --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 2 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm UQ --rip_per_scene_algorithm UQ --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 3 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm UQ --rip_per_scene_algorithm UQ --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping

# RIP mean mean
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 1 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm MA --rip_per_scene_algorithm MA --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 2 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm MA --rip_per_scene_algorithm MA --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints --model_name dim --data_use_prerendered True --torch_seed 3 --model_dim_hidden 512 --data_num_workers 10 --debug_overfit_eval True --debug_overfit_n_examples 512 --rip_per_plan_algorithm MA --rip_per_scene_algorithm MA --rip_k 3 --exp_batch_size 128 --wandb_project rip-prototyping

# Plot results
python plot_retention_curves.py --results_dir /scratch-ssd/neiand/yandex/metrics --plot_dir plots

# And grab the plots
rsync -rP neiand@oat4.cs.ox.ac.uk:/users/neiand/workspace/uncertainty-challenge/sdc_motion_prediction/plots .

## ** Experiments **

# Today, we are going to restrict to 5000 examples -- not enough time to get results on more. Should be sufficient (5000 examples ~ 100000 prediction requests)
# In-Domain: 5000 scenes, 75662 prediction requests
# OOD:
# * Exp 1: Ivan's models should probably outperform mine! *
# Fix to 3 models each
# RIP MA / MA

# Neil's models
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --model_name dim --data_use_prerendered True --torch_seed 1 --model_dim_hidden 512 --data_num_workers 10 --rip_per_plan_algorithm MA --rip_per_scene_algorithm MA --rip_k 3 --exp_batch_size 512 --wandb_project jun10-metric-exp1 --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints_neil --model_prefix Neil --debug_overfit_eval True --debug_overfit_n_examples 5000

# Ivan's models
python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --model_name dim --data_use_prerendered True --torch_seed 1 --model_dim_hidden 512 --data_num_workers 10 --rip_per_plan_algorithm MA --rip_per_scene_algorithm MA --rip_k 1 --exp_batch_size 512 --wandb_project jun10-metric-exp1 --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints_ivan --model_prefix Ivan --debug_overfit_eval True --debug_overfit_n_examples 5000

python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --model_name dim --data_use_prerendered True --torch_seed 1 --model_dim_hidden 512 --data_num_workers 10 --rip_per_plan_algorithm MA --rip_per_scene_algorithm MA --rip_k 3 --exp_batch_size 512 --wandb_project jun10-metric-exp1 --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints_ivan --model_prefix Ivan --debug_overfit_eval True --debug_overfit_n_examples 5000

python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --model_name dim --data_use_prerendered True --torch_seed 1 --model_dim_hidden 512 --data_num_workers 10 --rip_per_plan_algorithm MA --rip_per_scene_algorithm MA --rip_k 5 --exp_batch_size 512 --wandb_project jun10-metric-exp1 --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints_ivan --model_prefix Ivan --debug_overfit_eval True --debug_overfit_n_examples 5000


python plot_retention_curves.py --results_dir /scratch-ssd/neiand/yandex/metrics --plot_dir plots_jun10_exp1
