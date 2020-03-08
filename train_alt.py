from __future__ import print_function

import os
import sys
import random
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_gen import GoL_Sup_Dataset, OrderedGOLDataset

from utils import (AverageMeter, save_checkpoint)
from models import (BaselineCNN, CustomFeatureCNN, PatternEncoder, PatternDecoder)

if __name__ == '__main__':
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save checkpoints')
    parser.add_argument('data_order', type=str, help='ordered or mixed?')
    parser.add_argument('mode', type=str, help='what kind of model to run?')
    parser.add_argument('--img_size', type=int, default=16, help='board size?')
    # parser.add_argument('--custom', action='store_true', help='Using custom features?')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size [default=100]')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate [default=0.01]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs [default: 100]')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    args.out_dir = args.out_dir+"_"+args.data_order+"_"+args.mode
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(args)

    def train(args):
        # Define training dataset & build vocab
        train_datasets = get_datasets(args, 'Train')
        train_loaders = []
        for dataset in train_datasets:
            train_loaders.append(DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size))

        # Define test dataset
        val_dataset = get_datasets(args, 'Validation')
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        # Define model & optimizer
        if args.mode == 'baseline':
            conv_pred = BaselineCNN(img_size=args.img_size)
            optimizer = torch.optim.Adam(conv_pred.parameters(), lr=args.lr)
            conv_pred.to(device)
            models = [conv_pred]
        
        best_loss = float('inf')
        track_loss = np.zeros((args.epochs, 2))

        if args.data_order == 'mixed':
            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(epoch, models, optimizer, train_loaders[0])
                test_loss = test_one_epoch(epoch, models, test_loader)

                is_best = test_loss < best_loss
                best_loss = min(test_loss, best_loss)
                track_loss[epoch - 1, 0] = train_loss
                track_loss[epoch - 1, 1] = test_loss
                
                save_checkpoint({
                    'epoch': epoch,
                    'models': [model.state_dict() for model in models],
                    'optimizer': optimizer.state_dict(),
                    'track_loss': track_loss,
                    'cmd_line_args': args,
                }, is_best, folder=args.out_dir,
                filename='checkpoint')
                np.save(os.path.join(args.out_dir,
                    'loss.npy'), track_loss)
        elif args.data_order == 'ordered':
            for i in range(len(train_loaders)):
                print("Training on {i}th dataset")
                for epoch in range(1, args.epochs + 1):
                    train_loss = train_one_epoch(epoch, models, optimizer, train_loaders[i])
                    test_loss = test_one_epoch(epoch, models, test_loader)

                    is_best = test_loss < best_loss
                    best_loss = min(test_loss, best_loss)
                    track_loss[epoch - 1, 0] = train_loss
                    track_loss[epoch - 1, 1] = test_loss
                    
                    save_checkpoint({
                        'epoch': epoch+args.epochs*i,
                        'models': [model.state_dict() for model in models],
                        'optimizer': optimizer.state_dict(),
                        'track_loss': track_loss,
                        'cmd_line_args': args,
                    }, is_best, folder=args.out_dir,
                    filename='checkpoint')
                    np.save(os.path.join(args.out_dir,
                        'loss.npy'), track_loss)


    def train_one_epoch(epoch, models, optimizer, train_loader):
        for model in models:
            model.train()

        loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_loader))
        for batch_idx, (curr_pat, next_pat) in enumerate(train_loader):
            batch_size = curr_pat.size(0) 

            # breakpoint()
            curr_pat = curr_pat.float()
            next_pat = next_pat.float()
            # breakpoint()
            
            # obtain predicted pattern
            out = curr_pat
            for model in models:
                out = model(out)
                # breakpoint()
            pred_next = out

            # breakpoint()
            # loss: mean-squared error
            loss = F.mse_loss(pred_next, next_pat)

            # breakpoint()
            # train
            loss_meter.update(loss.item(), batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss_meter.avg})
            pbar.update()
        pbar.close()
            
        if epoch % 10 == 0:
            print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        
        return loss_meter.avg


    def test_one_epoch(epoch, models, test_loader):
        for model in models:
            model.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()

            pbar = tqdm(total=len(test_loader))
            for batch_idx, (curr_pat, next_pat) in enumerate(test_loader):
                batch_size = curr_pat.size(0) 

                # obtain predicted pattern
                out = curr_pat
                for model in models:
                    out = model(out)
                pred_next = out

                # loss: mean-squared error
                loss = F.mse_loss(pred_next, next_pat)

                loss_meter.update(loss.item(), batch_size)

                pbar.set_postfix({'loss': loss_meter.avg})
                pbar.update()
            pbar.close()
            if epoch % 10 == 0:
                print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
        return loss_meter.avg

    train(args)

def get_datasets(args, split):
    if split != 'Train':
        return OrderedGOLDataset(split=split)
    
    if args.data_order == 'ordered':
        datasets = []
        for data_type in ['still', 'oscillator', 'spaceship']:
            datasets.append(OrderedGOLDataset(split=split, data_type=data_type, data_order='ordered'))
        return datasets
    elif args.data_order == 'mixed':
        return [OrderedGOLDataset(split=split, data_order='mixed')]