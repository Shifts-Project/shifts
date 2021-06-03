import os
from collections import defaultdict
from functools import partial
from typing import Mapping

import torch
import torch.distributions as D
import torch.optim as optim
import tqdm as tq
from transformers import get_cosine_schedule_with_warmup

from sdc.dataset import load_datasets, load_dataloaders
from sdc.metrics import SDCLoss
from sdc.oatomobile.torch.baselines import batch_transform
from sdc.oatomobile.torch.baselines import init_model
from sdc.oatomobile.torch.savers import Checkpointer
from sdc.oatomobile.utils.loggers.wandb import WandbLogger
from sdc.oatomobile.torch.baselines.robust_imitative_planning import (
    load_rip_checkpoints)


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
    checkpoint_frequency = c.exp_checkpoint_frequency
    output_shape = c.model_output_shape
    num_timesteps_to_keep, _ = output_shape
    data_dtype = c.data_dtype
    device = c.exp_device
    model_name = c.model_name
    is_rip = (c.model_rip_algorithm is not None)
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

    # Create checkpoint dir, if necessary; init Checkpointer.
    checkpoint_dir = f'{c.dir_checkpoint}/{full_model_name}'

    if c.model_rip_algorithm:
        model = load_rip_checkpoints(
            model=model, device=device, k=c.model_rip_k,
            checkpoint_dir=checkpoint_dir)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpointer = Checkpointer(
            model=model, ckpt_dir=checkpoint_dir, torch_seed=c.torch_seed)

    # Init dataloaders.
    # Split = None loads train, validation, and test.
    datasets = load_datasets(c, splits=None)
    train_dataloader, eval_dataloaders = load_dataloaders(datasets, c)

    # Init object for computing loss and metrics.
    sdc_loss = SDCLoss()

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
    if model_name == 'dim':
        noise_level = c.dim_noise_level
        train_args['noise_level'] = noise_level

        # Theoretical limit of NLL, given the noise level.
        nll_limit = -torch.sum(  # pylint: disable=no-member
            D.MultivariateNormal(
                loc=torch.zeros(output_shape[-2] * output_shape[-1]),
                scale_tril=torch.eye(
                    output_shape[-2] * output_shape[-1]) * noise_level,
            ).log_prob(
                torch.zeros(output_shape[-2] * output_shape[-1])))

    # @profile
    def train_epoch(
            dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Performs an epoch of gradient descent optimization on `dataloader`."""
        model.train()
        loss = 0.0
        steps = 0
        with tq.tqdm(dataloader) as pbar:
            for batch in pbar:
                # Prepares the batch.
                batch = downsize_cast_batch_transform(batch)
                train_args['batch'] = batch

                # Performs a gradient-descent step.
                loss += train_step(**train_args)
                steps += 1

        return loss / steps, steps

    def evaluate_epoch(
      dataloader: torch.utils.data.DataLoader) -> Mapping[str, torch.Tensor]:
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
                    for key, value in loss_dict.items():
                        if key not in eval_loss_dict.keys():
                            eval_loss_dict[key] = value
                        else:
                            eval_loss_dict[key] += value

                steps += 1

        for key in eval_loss_dict:
            eval_loss_dict[key] /= steps

        return eval_loss_dict

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

    with tq.tqdm(range(num_epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            epoch_loss_dict = defaultdict(dict)

            if not is_rip:
                loss_train, epoch_steps = train_epoch(train_dataloader)
                epoch_loss_dict['train']['moscow__train'] = loss_train
                steps += epoch_steps
                # write(model, dataloader_train, writer, "train", loss_train, epoch)

            # Evaluates model on validation datasets
            for dataset_key, dataloader_val in validation_dataloaders.items():
                loss_val_dict = evaluate_epoch(dataloader_val)
                for loss_key, loss_value in loss_val_dict.items():
                    epoch_loss_dict[
                        'validation'][
                        f'{dataset_key}__{loss_key}'] = loss_value

            # write(model, dataloader_val, writer, "val", loss_val, epoch)

            # Checkpoints model weights.
            if (not is_rip) and (epoch % checkpoint_frequency == 0):
                checkpointer.save(epoch)

            # Updates progress bar description.
            if is_rip:
                pbar_string = ''
            else:
                pbar_string = (
                   'TL: {:.2f} | '.format(
                       epoch_loss_dict['train'][
                           'moscow__train'].detach().cpu().numpy().item()))

            if c.model_name == 'dim':
                pbar_string += 'THEORYMIN: {:.2f} | '.format(nll_limit)

            for dataset_key, loss_val in epoch_loss_dict['validation'].items():
                pbar_string += 'VL {} {:.2f} | '.format(
                    dataset_key, loss_val.detach().cpu().numpy().item())

            pbar_string += '\n'
            pbar_epoch.set_description(pbar_string)

            # Log to wandb
            logger.log(epoch_loss_dict, steps, epoch)

            if not is_rip:
                scheduler.step()

# def write(
#     model: Union[BehaviouralModel, ImitativeModel],
#     dataloader: torch.utils.data.DataLoader,
#     writer: TensorBoardLogger,
#     split: str,
#     loss: torch.Tensor,
#     epoch: int,
# ) -> None:
#     """Visualises model performance on `TensorBoard`."""
#
#
# # Gets a sample from the dataset.
# batch = next(iter(dataloader))
# # Prepares the batch.
# batch = transform(batch)
# # Generates predictions.
# with torch.no_grad():
#     predictions = model(**batch)
#
# # Logs on `TensorBoard`.
# writer.log(
#     split=split,
#     loss=loss.detach().cpu().numpy().item(),
#     overhead_features=batch["visual_features"].detach().cpu().numpy()[:8],
#     predictions=predictions.detach().cpu().numpy()[:8],
#     ground_truth=batch["player_future"].detach().cpu().numpy()[:8],
#     global_step=epoch,
# )

