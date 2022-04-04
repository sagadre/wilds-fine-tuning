import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloaders.cmip6 import Cmip6
from models.climate_nerf import ClimateNerf
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(model, device, train_loader, criterion, optimizer):

    model.train()
    train_loss = 0
    num_samples = 0

    for sample in train_loader:

        batch_size = sample['lat'].shape[0]
        num_samples += batch_size


        print(sample['lat'].shape)
        # put data on appropriate device
        input_data = torch.cat((sample['lat'].unsqueeze(-1), sample['long'].unsqueeze(-1), sample['time'].unsqueeze(-1)), 1).to(device)
        gt = sample['temp'].unsqueeze(-1).to(device)

        # zero out gradients
        optimizer.zero_grad()

        # forward pass
        pred = model(input_data)

        # compute loss
        loss = criterion(pred, gt)

        # sum up all the losses and ious
        train_loss += loss.item() * batch_size

        # back prop
        loss.backward()

        # increment learning schedule
        optimizer.step()

    train_loss /= num_samples

    return train_loss


def val(model, device, val_loader, criterion):
    """
    Similar to train(), but no need to backward and optimize.
    """
    model.eval()
    val_loss = 0.
    num_samples = 0

    with torch.no_grad():
        for sample in val_loader:

            # put data on appropriate device
            input_data = torch.cat((sample['lat'].unsqueeze(-1), sample['long'].unsqueeze(-1), sample['time'].unsqueeze(-1)), 1).to(device)
            gt = sample['temp'].unsqueeze(-1).to(device)

            # forward pass
            pred = model(input_data)

            # compute loss
            loss = criterion(pred, gt)

            # sum up all the losses
            val_loss += loss.item() * input_data.shape[0]
            num_samples += input_data.shape[0]

        val_loss /= num_samples

    return val_loss


def main(args):
    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Define directories
    train_json = f'{args.data_dir}/train.json'
    val_json = f'{args.data_dir}/val.json'
    test_json = f'{args.data_dir}/test.json'

    # Create Datasets. You can use check_dataset(your_dataset) to check your implementation.
    train_dataset = Cmip6(train_json)
    val_dataset = Cmip6(val_json)
    test_dataset = Cmip6(test_json)

    # Prepare Dataloaders. You can use check_dataloader(your_dataloader) to check your implementation.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    # Prepare model

    model = ClimateNerf()
    model.to(device)

    # Define criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, last_epoch=-1)


    # Train and validate the model
    # TODO: Remember to include the saved learning curve plot in your report
    epoch, max_epochs = 1, 20  # TODO: you may want to make changes here
    while epoch <= args.epochs:
        print(f'Epoch {epoch}')
        train_loss = train(model, device, train_loader, criterion, optimizer)
        print(train_loss)

        # TODO: save model
        epoch += 1
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', action='store', help='data dir')
    parser.add_argument('--val', action='store_true', help='run on val set')
    parser.add_argument('--test', action='store_true', help='run on test set')
    parser.add_argument('--batch-size', action='store', default=128, help='batch size for train or val/test')
    parser.add_argument('--n-workers', action='store', default=8, help='number of worker threads')
    parser.add_argument('--lr', action='store', default=0.01, help='starting learning rate')
    parser.add_argument('--epochs', action='store', default=100, help='starting learning rate')

    args = parser.parse_args()
    main(args)
