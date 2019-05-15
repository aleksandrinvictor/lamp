import numpy as np
from numpy import ndarray

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim
import os

from lamp.datasets import SimpleDataset

from tqdm import tqdm, tqdm_notebook
from typing import Iterator, Iterable, Union, Callable, Dict


OPTIMIZERS = {
    'adam': torch.optim.Adam
}

LOSS_FUNCS = {
    'mse': nn.MSELoss(),
    'mae': nn.L1Loss()
}

def _validate(
        validation_generator: Union[Iterator, Iterable],
        model: nn.Module,
        criterion,
        device: torch.device
) -> float:

    validation_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(validation_generator):
            val_inputs, val_labels = batch

            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_labels)

            validation_loss += loss.item()

    return validation_loss / len(validation_generator)


def _train(
        model: nn.Module,
        train_generator: Union[Iterator, Iterable],
        validation_generator: Union[Iterator, Iterable],
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        early_stopping_rounds: int = np.inf,
        print_every: int = 10,
        start_epoch: int = 0,
        max_epochs: int = 100,
        checkpoints_path: str = None,
        inline: bool = False
):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = model.to(device)
    model.train()

    early_stopping = EarlyStopping(
        patience=early_stopping_rounds,
        min_delta=0,
        checkpoints_path=checkpoints_path
    )
    running_loss = 0.0
    running_loss_epoch = 0.0

    hist = {
        'train_loss': [],
        'val_loss': []
    }

    tqdm_func = tqdm_notebook if inline else tqdm
    # Loop over epochs
    for epoch in tqdm_func(range(start_epoch, max_epochs), desc='epochs done'):

        # Training
        for i, batch in enumerate(train_generator):
            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_epoch += loss.item()
            if (i + 1) % print_every == 0:  # print every print_every mini-batches

                print(f'[epoch: {epoch + 1}, batch: {i + 1}] \
                 train loss: {running_loss / print_every}')
                running_loss = 0.0

        model.eval()
        validation_loss = _validate(
            validation_generator,
            model,
            criterion,
            device
        )
        model.train()

        print(
            f"[epoch: {epoch + 1}] train loss: {running_loss_epoch / len(train_generator)} \
                    val loss: {validation_loss}"
        )

        hist['train_loss'].append(running_loss_epoch / len(train_generator))
        hist['val_loss'].append(validation_loss)

        if early_stopping(
            validation_loss, 
            model, 
            optimizer, 
            epoch
        ):
            print(f'Early stopping. stop epoch: {early_stopping.stop_epoch + 1}, best score: {early_stopping.best_score}')
            break

        running_loss_epoch = 0
    
    return model, hist


def train_model(
    model: nn.Module,
    train_X: ndarray,
    train_y: ndarray,
    val_X: ndarray,
    val_y: ndarray,
    batch_size: int = 64,
    max_epochs: int = 100,
    loss: str = 'mse',
    optimizer: str = 'adam',
    learning_rate: float = 0.01,
    early_stopping_rounds: int = 5,
    print_every: int = np.inf,
    checkpoint: Dict = None,
    checkpoints_path: str = None,
    inline: bool = False
) -> Union[nn.Module, Dict]:

    train_dataset = SimpleDataset(train_X, train_y)
    validation_dataset = SimpleDataset(val_X, val_y)

    train_generator = DataLoader(
        train_dataset,
        batch_size,
        num_workers=3,
        shuffle=False
    )
    validation_generator = DataLoader(
        validation_dataset,
        batch_size,
        num_workers=3,
        shuffle=False
    )

    optimizer = OPTIMIZERS[optimizer](
        model.parameters(),
        lr=learning_rate
    )

    loss_fn = LOSS_FUNCS[loss]

    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        # best_val_loss = checkpoint['best_val_loss']
        optimizer_dict = checkpoint['optimizer']

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(optimizer_dict)
    else:
        start_epoch = 0
    
    model, hist = _train(
        model,
        train_generator,
        validation_generator,
        loss_fn,
        optimizer,
        early_stopping_rounds,
        print_every,
        start_epoch,
        max_epochs,
        checkpoints_path,
        inline
    )

    return model, hist


class EarlyStopping(object):

    def __init__(
            self,
            patience: int = 1,
            min_delta: float = 0,
            checkpoints_path: str = None
    ):
        self.counter = 0
        self.min_delta = min_delta
        self.patience = patience
        self.best_score = None
        self.stop_epoch = None
        self.checkpoints_path = checkpoints_path

    def __call__(
            self,
            score,
            model,
            optimizer,
            epoch
    ):

        if not self.best_score:
            self.best_score = score
            self.stop_epoch = epoch

        if self.best_score - score <= self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.stop_epoch = epoch
            self.counter = 0
            if self.checkpoints_path is not None:
                self.__save_checkpoint(score, model, optimizer, epoch)

        return False

    def __save_checkpoint(
            self,
            score,
            model,
            optimizer,
            epoch
    ):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_val_loss': score,
            'optimizer': optimizer.state_dict()
        }
        filepath = os.path.join(self.checkpoints_path, f'checkpoint_epoch_{epoch + 1}.pth.tar')

        torch.save(state, filepath)
