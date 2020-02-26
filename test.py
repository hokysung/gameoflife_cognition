import os
import sys
import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data_gen import GoL_Sup_Dataset

from utils import (AverageMeter, save_checkpoint)
from models import (BaselineCNN, PatternEncoder, PatternDecoder)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    parser.add_argument('out_dir', type=str, help='where to store results to')
    parser.add_argument('data_type', type=str, help='random or pattern?')
    parser.add_argument('mode', type=str, help='what kind of model to run?')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='number of total iterations performed on each setting [default: 1]')
    parser.add_argument('--context_condition', type=str, default='all',
                        help='whether the dataset is to include all data')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # set learning device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.load_dir = args.load_dir+"_"+args.data_type+"_"+args.mode
    args.out_dir = args.load_dir+"_"+args.data_type+"_"+args.mode

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    def test_loss(models):
        '''
        Test model on newly seen dataset -- gives final test loss
        '''
        print("Computing final test loss on newly seen dataset...")
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

    def load_checkpoint(folder='./', filename='model_best'):
        print("\nloading checkpoint file: {}.pth.tar ...\n".format(filename)) 
        checkpoint = torch.load(os.path.join(folder, filename + '.pth.tar'))
        epoch = checkpoint['epoch']
        track_loss = checkpoint['track_loss']
        models = checkpoint['models']
        return epoch, track_loss, models

    print("\n=== begin testing ===")
    
    print("loading checkpoint ...")
    epoch, track_loss, models = \
        load_checkpoint(folder=args.load_dir,
                        filename='checkpoint_best')

    test_dataset = GoL_Sup_Dataset(types=args.data_type, split='Test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)

    if args.mode == 'baseline':
        conv_pred = BaselineCNN(img_size=30)
        conv_pred.load_state_dict(models[0])
        models = [conv_pred]
    else:
        pattern_enc = PatternEncoder(img_size=30)
        pattern_dec = PatternDecoder(img_size=30)
        pattern_enc.load_state_dict(models[0])
        pattern_dec.load_state_dict(models[1])
        models = [pattern_enc, pattern_dec]

    loss = test_loss(models)
    print()
    print("Final test loss: {}".format(loss))