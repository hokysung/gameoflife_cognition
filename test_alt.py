import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchnet import meter

from torch.utils.data import DataLoader
from data_gen import GoL_Sup_Dataset, OrderedGOLDataset

from utils import (AverageMeter, save_checkpoint)
from models import (BaselineCNN, CustomFeatureCNN, PatternEncoder, PatternDecoder)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', type=str, help='where to load checkpoints from')
    # parser.add_argument('out_dir', type=str, help='where to store results to')
    # parser.add_argument('train_data_type', type=str, help='random or pattern?')
    # parser.add_argument('test_data_type', type=str, help='random or pattern?')
    parser.add_argument('mode', type=str, help='what kind of model to run?')
    parser.add_argument('--img_size', type=int, default=30, help='board size?')
    parser.add_argument('--custom', action='store_true', help='Using custom features?')
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

    # args.load_dir = args.load_dir
    #args.out_dir = args.load_dir+"_"+args.data_type+"_"+args.mode

    #if not os.path.isdir(args.out_dir):
    #    os.makedirs(args.out_dir)

    def matthews(pred, truth):
        matthews_val = 0
        for i in range(len(pred)):
            confusion_matrix = meter.ConfusionMeter(2)
            confusion_matrix.add(pred[i].view(-1), truth[i].view(-1))
            tp = int(confusion_matrix.conf[0][0])
            fp = int(confusion_matrix.conf[0][1])
            fn = int(confusion_matrix.conf[1][0])
            tn = int(confusion_matrix.conf[1][1])
            try:
                m = float((tp * tn) - (fp * fn)) / \
                    math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            except ZeroDivfnisionError:
                m = 0

            matthews_val += m
            # breakpoint()
            
        return matthews_val / len(pred)

    def test_loss(models, epoch, test_loader):
        '''
        Test model on newly seen dataset -- gives final test loss
        '''
        print("Computing final test loss on newly seen dataset...")
        for model in models:
            model.eval()

        with torch.no_grad():
            loss_meter = AverageMeter()
            matthews_meter = AverageMeter()

            pbar = tqdm(total=len(test_loader))
            for batch_idx, (curr_pat, next_pat) in enumerate(test_loader):
                batch_size = curr_pat.size(0) 

                # obtain predicted pattern
                out = curr_pat
                for model in models:
                    out = model(out)
                pred_next = out

                # breakpoint()

                if not os.path.exists('./images'):
                    os.makedirs('./images')
                for j in range(10):
                    original = curr_pat[j].detach().numpy()
                    #plt.plot(original, cmap="Greys", interpolation='nearest')
                    plt.imsave('./images/trueprev_{}_{}.png'.format(iteration, j), original,cmap="Greys")
                    answer = next_pat[j].detach().numpy()
                    plt.imsave('./images/truenext_{}_{}.png'.format(iteration, j), answer, cmap="Greys")
                    prediction = pred_next[j].detach().numpy()
                    plt.imsave('./images/prediction_{}_{}.png'.format(iteration, j), prediction, cmap="Greys")
                

                #matthews
                m = matthews((pred_next>0.5).float(), next_pat)
                # loss: mean-squared error
                loss = F.mse_loss(pred_next, next_pat)

                matthews_meter.update(m, batch_size)
                loss_meter.update(loss.item(), batch_size)

                pbar.set_postfix({'loss': loss_meter.avg})
                pbar.update()
            pbar.close()
            print('====> Test Epoch: {}\tLoss: {:.4f}\tMatthews:{:.4f}'.format(epoch, loss_meter.avg, matthews_meter.avg))
        return loss_meter.avg, matthews_meter.avg

    def load_checkpoint(folder='./', filename='model_best'):
        print("\nloading checkpoint file: {}.pth.tar ...\n".format(os.path.join(folder,filename))) 
        # breakpoint()
        checkpoint = torch.load(os.path.join(folder, filename + '.pth.tar'))
        epoch = checkpoint['epoch']
        track_loss = checkpoint['track_loss']
        models = checkpoint['models']
        seed = checkpoint['seed']
        return epoch, track_loss, models, seed

    print("\n=== begin testing ===")

    def test(iteration, args):
        print("loading checkpoint ...")
        load_dir = os.path.join(args.load_dir, 'run_{}'.format(iteration))
        epoch, track_loss, models, seed = \
            load_checkpoint(folder=load_dir,
                            filename='checkpoint_3_best')

        # Define test dataset
        test_dataset = OrderedGOLDataset(split='Test')
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)

        if args.mode == 'baseline':
            conv_pred = BaselineCNN(img_size=args.img_size)
            conv_pred.load_state_dict(models[0])
            models = [conv_pred]
        else:
            pattern_enc = PatternEncoder(img_size=args.img_size)
            pattern_dec = PatternDecoder(img_size=args.img_size)
            pattern_enc.load_state_dict(models[0])
            pattern_dec.load_state_dict(models[1])
            models = [pattern_enc, pattern_dec]

        loss, matthews = test_loss(models, epoch, test_loader)
        print()
        print("Random seed was: {}".format(seed))
        print("Final test loss: {}".format(loss))
        print("Final matthews score: {}".format(matthews))

        # print("Saving to {}".format())
        np.save(os.path.join(load_dir, 'test_res.npy'), np.array([loss, matthews]))

    for iteration in range(10):
        print()
        print("Test for Iteration {}".format(iteration))
        
        test(iteration, args)