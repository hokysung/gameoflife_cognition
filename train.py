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
from data_gen import GoL_Sup_Dataset

from utils import (AverageMeter, save_checkpoint)
from models import (BaselineCNN, CustomFeatureCNN, PatternEncoder, PatternDecoder)

if __name__ == '__main__':
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save checkpoints')
    parser.add_argument('data_type', type=str, help='random or pattern?')
    parser.add_argument('mode', type=str, help='what kind of model to run?')
    parser.add_argument('--img_size', type=int, default=30, help='board size?')
    parser.add_argument('--custom', action='store_true', help='Using custom features?')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size [default=100]')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate [default=0.01]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs [default: 100]')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    args.out_dir = args.out_dir+"_"+args.data_type+"_"+args.mode
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(args)

    def train(args):
        # Define training dataset & build vocab
        train_dataset = GoL_Sup_Dataset(data_type=args.data_type, custom_features=args.custom, split='Train')
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        N_mini_batches = len(train_loader)

        # Define test dataset
        test_dataset = GoL_Sup_Dataset(data_type=args.data_type, custom_features=args.custom, split='Validation')
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        # Define model & optimizer
        if args.mode == 'baseline':
            conv_pred = BaselineCNN(img_size=args.img_size) if not args.custom else CustomFeatureCNN(img_size=args.img_size)
            optimizer = torch.optim.Adam(conv_pred.parameters(), lr=args.lr)
            conv_pred.to(device)
            models = [conv_pred]
        elif args.mode == 'autoencoder':
            pattern_enc = PatternEncoder(img_size=args.img_size)
            pattern_dec = PatternDecoder(img_size=args.img_size)
            optimizer = torch.optim.Adam(
                chain(
                pattern_enc.parameters(),
                pattern_dec.parameters(),
            ), lr=args.lr)
            # models = [pattern_enc]
            models = [pattern_enc, pattern_dec]
        
        best_loss = float('inf')
        track_loss = np.zeros((args.epochs, 2))

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(epoch, models, optimizer, train_loader)
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