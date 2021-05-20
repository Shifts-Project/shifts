import os
from collections import defaultdict
from functools import partial

import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
import tqdm as tq

from transformers import get_cosine_schedule_with_warmup
from sdc.dataset import load_datasets, load_dataloaders
from sdc.oatomobile.torch.baselines import (
    MODEL_TO_FULL_NAME, MODEL_NAME_TO_CLASS_FNS)
from sdc.oatomobile.torch.baselines import batch_transform
from sdc.oatomobile.torch.savers import Checkpointer
from sdc.oatomobile.utils.loggers.wandb import WandbLogger


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        # do stuff here?
        self.values = torch.randn((50000, 100, 100, 16))
        self.labels = torch.randn((50000, 2, 25))
        # self.labels = labels

    def __len__(self):
        return len(self.values)  # number of samples in the dataset

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


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
    num_timesteps_to_keep = 25
    downsample_hw = (c.exp_image_downsize_hw, c.exp_image_downsize_hw)
    data_dtype = c.data_dtype
    device = c.exp_device
    dim_hidden = c.model_dim_hidden

    downsize_cast_batch_transform = partial(
        batch_transform, device=device, downsample_hw=downsample_hw,
        dtype=data_dtype, num_timesteps_to_keep=num_timesteps_to_keep)
    output_shape = (num_timesteps_to_keep, 2)  # Predict 25 timesteps
    # in_channels = 9
    in_channels = 16  # TODO: speed up featurization of roads/HD map

    # Initializes the model and its optimizer.

    # Obtain model class (e.g., ImitativeModel or BehavioralModel)
    # and its respective train/evaluate steps.
    model_name = c.model_name
    print(f'Training {MODEL_TO_FULL_NAME[model_name]}.')
    model_class, train_step, evaluate_step = (
        MODEL_NAME_TO_CLASS_FNS[c.model_name])
    model = model_class(
        in_channels=in_channels, dim_hidden=dim_hidden,
        output_shape=output_shape).to(device=device)
    print(f'Model has {count_parameters(model)} parameters,'
          f'batch size {c.exp_batch_size}.')
    criterion = nn.L1Loss(reduction="none")
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    # Based on the fairseq implementation, which is based on BERT
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=num_epochs)

    # Create checkpoint dir, if necessary; init Checkpointer.
    checkpoint_dir = c.dir_checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = Checkpointer(model=model, ckpt_dir=checkpoint_dir)

    # Init dataloaders.
    # Split = None loads train, validation, and test.
    datasets = load_datasets(c, split=None)
    train_dataloader, eval_dataloaders = load_dataloaders(datasets, c)

    # Init train and evaluate args for respective model backbone.
    train_args = {
        'model': model,
        'optimizer': optimizer,
        'clip': clip_gradients
    }
    evaluate_args = {'model': model}
    if model_name == 'bc':
        train_args['criterion'] = criterion
        evaluate_args['criterion'] = criterion
    elif model_name == 'dim':
        noise_level = c.dim_noise_level
        train_args['noise_level'] = noise_level

        # Theoretical limit of NLL.
        nll_limit = -torch.sum(  # pylint: disable=no-member
            D.MultivariateNormal(
                loc=torch.zeros(output_shape[-2] * output_shape[-1]),
                # pylint: disable=no-member
                scale_tril=torch.eye(output_shape[-2] * output_shape[
                    -1]) *  # pylint: disable=no-member
                           noise_level,  # pylint: disable=no-member
            ).log_prob(torch.zeros(output_shape[-2] * output_shape[
                -1])))  # pylint: disable=no-member

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
      dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Performs an evaluation of the `model` on the `dataloader."""
        model.eval()
        loss = 0.0
        steps = 0
        with tq.tqdm(dataloader) as pbar:
            for batch in pbar:
                # Prepares the batch.
                batch = downsize_cast_batch_transform(batch)
                evaluate_args['batch'] = batch

                # Accumulates loss in dataset.
                with torch.no_grad():
                    loss += evaluate_step(**evaluate_args)

                steps += 1

        return loss / steps

    # Initialize wandb logger state
    logger = WandbLogger(optimizer)
    logger.start_counting()
    steps = 0
    if c.debug_overfit_test_data_only:
        validation_dataloaders = {}
    else:
        validation_dataloaders = eval_dataloaders['validation']

    with tq.tqdm(range(num_epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            # Trains model on whole training dataset
            epoch_loss_dict = defaultdict(dict)

            loss_train, epoch_steps = train_epoch(train_dataloader)
            epoch_loss_dict['train']['moscow__train'] = loss_train
            steps += epoch_steps
            # write(model, dataloader_train, writer, "train", loss_train, epoch)

            # Evaluates model on validation datasets
            for dataset_key, dataloader_val in validation_dataloaders.items():
                loss_val = evaluate_epoch(dataloader_val)
                epoch_loss_dict['validation'][dataset_key] = loss_val

            # write(model, dataloader_val, writer, "val", loss_val, epoch)

            # Checkpoints model weights.
            if epoch % checkpoint_frequency == 0:
                checkpointer.save(epoch)

            # Updates progress bar description.
            pbar_string = (
               'TL: {:.2f} | '.format(
                   epoch_loss_dict['train'][
                       'moscow__train'].detach().cpu().numpy().item()))

            if c.model_name == 'dim':
                pbar_string += 'THEORYMIN: {:.2f}'.format(nll_limit)

            for dataset_key, loss_val in epoch_loss_dict['validation'].items():
                pbar_string += 'VL {} {:.2f} | '.format(
                    dataset_key, loss_val.detach().cpu().numpy().item())

            pbar_string += '\n'
            pbar_epoch.set_description(pbar_string)

            # Log to wandb
            logger.log(epoch_loss_dict, steps, epoch)

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

