import argparse


def build_parser():
    """Build parser."""
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    ###########################################################################
    # #### Wandb ##############################################################
    ###########################################################################
    parser.add_argument(
        '--wandb_project', type=str, default='sdc',
        help='Wandb project name.')

    ###########################################################################
    # #### Directories ########################################################
    ###########################################################################
    parser.add_argument(
        '--dir_wandb', type=str, default='./',
        help='Directory to which wandb logs.')
    parser.add_argument(
        '--dir_data', type=str, default='data',
        help='Directory where SDC data is stored. We also use this to cache '
             'torch hub models.')
    parser.add_argument(
        '--dir_checkpoint', type=str, default='model_checkpoints',
        help='Directory to which model checkpoints are stored.')

    ###########################################################################
    # #### General ############################################################
    ###########################################################################
    parser.add_argument(
        '--verbose', type='bool', default=False,
        help='Extra logging, sanity checks.')
    parser.add_argument(
        '--np_seed', type=int, default=42,
        help='NumPy seed (data processing, train/test splits, etc.).')
    parser.add_argument(
        '--torch_seed', type=int, default=42,
        help='Model seed.')

    ###########################################################################
    # #### Data ###############################################################
    ###########################################################################
    parser.add_argument(
        '--data_num_workers', type=int, default=4,
        help='Number of workers to use in PyTorch data loading.')
    parser.add_argument(
        '--data_dtype',
        default='float32',
        type=str, help='Data type (supported for float32, float64) '
                         'used for data (e.g. inputs, ground truth).')

    ###########################################################################
    # #### Experiment #########################################################
    ###########################################################################
    parser.add_argument(
        '--exp_sweep', type='bool', default=False,
        help=f'Run an experiment sweep. Will clear out previous wandb logs.')
    parser.add_argument(
        '--exp_name', type=str, default=None,
        help=f'Specify an explicit name for the experiment - otherwise we '
             f'generate a unique wandb ID.')
    parser.add_argument(
        '--exp_group', type=str, default=None,
        help=f'Specify an explicit name for the experiment group.')
    parser.add_argument(
        '--exp_use_cuda', type='bool', default=True,
        help=f'Attempt to use CUDA device for training.')
    parser.add_argument(
        '--exp_print_every_nth_forward', dest='exp_print_every_nth_forward',
        default=False, type=int,
        help='Print during mini-batch as well for large epochs.')
    parser.add_argument(
        '--exp_lr', type=float, default=1e-3)
    parser.add_argument(
        '--exp_num_lr_warmup_epochs', type=int, default=10)
    parser.add_argument(
        '--exp_batch_size', type=int, default=16)
    parser.add_argument(
        '--exp_num_epochs', type=int, default=100)
    parser.add_argument(
        '--exp_image_downsize_hw', type=int, default=100,
        help='Downsize image input to '
             '(exp_image_downsize, exp_image_downsize).')
    parser.add_argument(
        '--exp_checkpoint_frequency', type=int, default=25,
        help='Model checkpoint frequency.')

    ###########################################################################
    # #### Model ##############################################################
    ###########################################################################

    parser.add_argument(
        '--model_name', type=str, default='bc',
        help="The backbone model. See "
             "sdc/oatomobile/torch/baselines/__init__.py.")
    parser.add_argument(
        '--model_dim_hidden', type=int, default=128,
        help="Number of hidden dims, generally the size "
             "of the embedding passed from the vision model "
             "(e.g., MobileNetV2) to the autoregressive "
             "decoder.")
    parser.add_argument('--model_weight_decay', type=float, default=0.0,
                        help="The L2 penalty (regularization) coefficient.")
    parser.add_argument(
        '--model_clip_gradients', type='bool', default=False,
        help='Clips gradients to 1.0.')

    ###########################################################################
    # #### Method-Specific Hypers #############################################
    ###########################################################################

    parser.add_argument('--dim_noise_level', type=float, default=1e-2,
                        help="")

    ###########################################################################
    # #### Debug ##############################################################
    ###########################################################################

    parser.add_argument(
        '--debug_overfit_eval', type='bool', default=False,
        help='Train on a very small subset, try to overfit.')
    parser.add_argument(
        '--debug_overfit_test_data_only', type='bool', default=False,
        help='If True, only use the test data.')
    parser.add_argument(
        '--debug_overfit_n_examples', type=int, default=10,
        help='Size of the subset of train on which we try to overfit.')


    return parser


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse/36031646"""
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")
