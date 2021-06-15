import os
from collections import defaultdict
from functools import partial
from typing import Mapping, Union, Text

import torch
import torch.optim as optim
import tqdm as tq
from transformers import get_cosine_schedule_with_warmup

from sdc.dataset import load_datasets, load_dataloaders
from sdc.metrics import SDCLoss
from sdc.oatomobile.tf.loggers import TensorBoardLogger
from sdc.oatomobile.torch.baselines import (
    ImitativeModel, BehaviouralModel, batch_transform, init_model)
from sdc.oatomobile.torch.baselines.robust_imitative_planning import (
    load_rip_checkpoints)
from sdc.oatomobile.torch.savers import Checkpointer
from sdc.oatomobile.utils.loggers.wandb import WandbLogger


def count_parameters(model):
    r"""
    Due to Federico Baldassarre
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# @profile
def train(c):
    # Retrieve config args.
    lr = c.exp_lr  # Learning rate
    weight_decay = c.model_weight_decay
    clip_gradients = c.model_clip_gradients
    num_epochs = c.exp_num_epochs
    num_warmup_epochs = c.exp_num_lr_warmup_epochs
    # checkpoint_frequency = c.exp_checkpoint_frequency
    output_shape = c.model_output_shape
    num_timesteps_to_keep, _ = output_shape
    data_dtype = c.data_dtype
    device = c.exp_device
    is_rip = (c.rip_per_plan_algorithm is not None and
              c.rip_per_scene_algorithm is not None)
    if c.exp_image_downsize_hw is None:
        downsample_hw = None
    else:
        downsample_hw = (c.exp_image_downsize_hw, c.exp_image_downsize_hw)

    downsize_cast_batch_transform = partial(
        batch_transform, device=device, downsample_hw=downsample_hw,
        dtype=data_dtype, num_timesteps_to_keep=num_timesteps_to_keep,
        data_use_prerendered=c.data_use_prerendered)

    # Obtain model backbone (e.g., ImitativeModel or BehavioralModel),
    # RIP wrapper if specified (e.g., model averaging),
    # and its respective train/evaluate steps.
    model, full_model_name, train_step, evaluate_step = init_model(c)

    if is_rip:
        # Should just be loading trained ensemble members
        # from checkpoint for RIP.
        optimizer, scheduler = None, None
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_epochs,
            num_training_steps=num_epochs)

    if c.model_checkpoint_key is not None:
        checkpoint_dir = f'{c.dir_checkpoint}/{c.model_checkpoint_key}'
    else:
        # Create checkpoint dir, if necessary; init Checkpointer.
        checkpoint_dir = f'{c.dir_checkpoint}/{full_model_name}'

    if is_rip:
        model = load_rip_checkpoints(
            model=model, device=device, k=c.rip_k,
            checkpoint_dir=checkpoint_dir)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpointer = Checkpointer(
            model=model, ckpt_dir=checkpoint_dir, torch_seed=c.torch_seed,
            checkpoint_frequency=c.exp_checkpoint_frequency)

    # Init dataloaders.
    # Split = None loads train, validation, and test.
    datasets = load_datasets(c, splits=None)
    train_dataloader, eval_dataloaders = load_dataloaders(datasets, c)

    # Init object for computing loss and metrics.
    sdc_loss = SDCLoss(full_model_name=full_model_name, c=c)

    # Init train and evaluate args for respective model backbone.
    train_args = {
        'model': model,
        'optimizer': optimizer,
        'clip': clip_gradients,
        'sdc_loss': sdc_loss
    }
    evaluate_args = {
        'model': model,
        'sdc_loss': sdc_loss
    }

    if c.tb_logging:
        dataset_names = ['moscow__train'] + list(
            set(eval_dataloaders['validation'].keys()))
        # TODO: remove
        # RIP eval on test set
        if is_rip:
            dataset_names += list(set(eval_dataloaders['test'].keys()))
        print(f'TensorBoard logging for datasets {dataset_names}.')
        writer = TensorBoardLogger(
            log_dir=c.dir_tensorboard, dataset_names=dataset_names)

    if is_rip:
        evaluate_args['cache_full_results'] = c.rip_cache_full_results

    # if model_name == 'dim':
        # noise_level = c.dim_noise_level
        # train_args['noise_level'] = noise_level

    # # Theoretical limit of NLL, given the noise level.
    # nll_limit = -torch.sum(  # pylint: disable=no-member
    #     D.MultivariateNormal(
    #         loc=torch.zeros(output_shape[-2] * output_shape[-1]),
    #         scale_tril=torch.eye(
    #             # output_shape[-2] * output_shape[-1]) * noise_level,
    #     ).log_prob(
    #         torch.zeros(output_shape[-2] * output_shape[-1])))

    # @profile
    def train_epoch(
            dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Performs an epoch of gradient descent optimization on `dataloader`."""
        model.train()
        train_loss_dict = {}
        loss = 0.0
        steps = 0
        with tq.tqdm(dataloader) as pbar:
            for batch in pbar:
                # Prepares the batch.
                batch = downsize_cast_batch_transform(batch)
                train_args['batch'] = batch

                # Performs a gradient-descent step.
                loss_dict = train_step(**train_args)
                for key, value in loss_dict.items():
                    if key not in train_loss_dict.keys():
                        train_loss_dict[key] = value
                    else:
                        train_loss_dict[key] += value

                steps += 1

        for key in train_loss_dict:
            train_loss_dict[key] /= steps

        return train_loss_dict, steps

    def evaluate_epoch(
      dataloader: torch.utils.data.DataLoader,
      dataset_key: str = None
    ) -> Mapping[str, torch.Tensor]:
        """Performs an evaluation of the `model` on the `dataloader."""
        model.eval()
        eval_loss_dict = {}
        steps = 0
        with tq.tqdm(dataloader) as pbar:
            for batch in pbar:
                # Prepares the batch.
                batch = downsize_cast_batch_transform(batch)
                evaluate_args['batch'] = batch

                # Accumulates loss in dataset.
                with torch.no_grad():
                    loss_dict = evaluate_step(**evaluate_args)

                    if loss_dict is not None:
                        for key, value in loss_dict.items():
                            if key not in eval_loss_dict.keys():
                                eval_loss_dict[key] = value
                            else:
                                eval_loss_dict[key] += value

                steps += 1

        if eval_loss_dict is not None:
            for key in eval_loss_dict:
                eval_loss_dict[key] /= steps
        if is_rip:
            eval_loss_dict = sdc_loss.evaluate_dataset_losses(dataset_key)

        return eval_loss_dict

    def write(
        model: Union[BehaviouralModel, ImitativeModel],
        dataloader: torch.utils.data.DataLoader,
        writer: TensorBoardLogger,
        dataset_name: str,
        loss_dict: Mapping[Text, torch.Tensor],
        epoch: int,
    ) -> None:
        """Visualises model performance on `TensorBoard`."""
        # Gets a sample from the dataset.
        batch = next(iter(dataloader))

        # Prepares the batch.
        batch = downsize_cast_batch_transform(batch)

        # Generates predictions.
        with torch.no_grad():
            predictions = model(**batch)

        if isinstance(model, BehaviouralModel):
            predictions, _ = predictions

        # Logs on `TensorBoard`.
        writer.log(
            dataset_name=dataset_name,
            loss_dict=loss_dict,
            overhead_features=batch["feature_maps"].detach().cpu().numpy()[:8],
            predictions=predictions.detach().cpu().numpy()[:8],
            ground_truth=batch[
                "ground_truth_trajectory"].detach().cpu().numpy()[:8],
            global_step=epoch,
        )

    # Initialize wandb logger state
    logger = WandbLogger(optimizer)
    logger.start_counting()
    steps = 0
    if c.debug_overfit_test_data_only:
        validation_dataloaders = {}
    else:
        try:
            validation_dataloaders = eval_dataloaders['validation']
        except KeyError:
            print('No validation sets found. Computing per-epoch test loss:')
            validation_dataloaders = eval_dataloaders['test']
            print(validation_dataloaders.keys())

    if is_rip:
        print('Running RIP evaluation. Setting num_epochs to 1.')
        num_epochs = 1
        test_dataloaders = eval_dataloaders['test']

    with tq.tqdm(range(num_epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            epoch_loss_dict = defaultdict(dict)

            if not is_rip:
                loss_train_dict, epoch_steps = train_epoch(train_dataloader)
                dataset_key = 'moscow__train'
                for loss_key, loss_value in loss_train_dict.items():
                    epoch_loss_dict['train'][
                        f'{dataset_key}__{loss_key}'] = (
                        loss_value.detach().cpu().numpy().item())
                steps += epoch_steps
                if c.tb_logging:
                    write(model, train_dataloader, writer, dataset_key,
                          loss_train_dict, epoch)

            # Evaluates model on validation datasets
            for dataset_key, dataloader_val in validation_dataloaders.items():
                loss_val_dict = evaluate_epoch(dataloader_val, dataset_key)
                for loss_key, loss_value in loss_val_dict.items():
                    epoch_loss_dict[
                        'validation'][
                        f'{dataset_key}__{loss_key}'] = (
                        loss_value.detach().cpu().numpy().item())

                if c.tb_logging:
                    write(model, dataloader_val, writer, dataset_key,
                        loss_val_dict, epoch)

            if is_rip:
                # TODO: remove in final code
                # For RIP, evaluate on the test sets.
                for dataset_key, dataloader_test in test_dataloaders.items():
                    loss_test_dict = evaluate_epoch(
                        dataloader_test, dataset_key)
                    for loss_key, loss_value in loss_test_dict.items():
                        epoch_loss_dict[
                            'test'][f'{dataset_key}__{loss_key}'] = (
                            loss_value.detach().cpu().numpy().item())
                    if c.tb_logging:
                        write(model, dataloader_test, writer, dataset_key,
                              loss_test_dict, epoch)
            else:
                # Checkpoints model weights if c.exp_checkpoint_validation_loss
                # has improved since last checkpoint.
                checkpointer.save(
                    epoch, epoch_loss_dict[
                        'validation'][c.exp_checkpoint_validation_loss])

            # Updates progress bar description.
            pbar_string = ''

            if not is_rip:
                pbar_string += 'Train Losses: '
                for dataset_key, loss_train in epoch_loss_dict['train'].items():
                    pbar_string += '{} {:.2f} | '.format(
                        dataset_key, loss_train)

            # if c.model_name == 'dim':
            #     pbar_string += 'THEORYMIN: {:.2f} | '.format(nll_limit)

            pbar_string += 'Val Losses: '
            for dataset_key, loss_val in epoch_loss_dict['validation'].items():
                pbar_string += '{} {:.2f} | '.format(dataset_key, loss_val)

            if is_rip:
                pbar_string += 'Test Losses: '
                for dataset_key, loss_test in epoch_loss_dict['test'].items():
                    pbar_string += '{} {:.2f} | '.format(
                        dataset_key, loss_test)

            pbar_string += '\n'
            pbar_epoch.set_description(pbar_string)

            # Log to wandb
            logger.log(epoch_loss_dict, steps, epoch)

            if not is_rip:
                scheduler.step()
