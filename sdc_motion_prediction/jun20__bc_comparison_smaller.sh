# BC Deterministic

sbatch run_sdc.sh python run.py --dir_wandb=/scratch-ssd/neiand/sdc/ --dir_data=/scratch-ssd/neiand/yandex --dir_checkpoint=/scratch-ssd/neiand/sdc/model_checkpoints_jun20_no_teacher_forcing --exp_batch_size 512 --exp_num_epochs 100000 --exp_checkpoint_frequency 1 --model_name bc --model_dim_hidden 512 --data_use_prerendered True --torch_seed 1 --data_num_workers 10 --wandb_project jun18-bc-dim-full-data --debug_bc_teacher_forcing False
