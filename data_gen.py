import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn.functional as F

import numpy as np
import torch
import torch.utils.data as data
import pickle

import os

import boardmaker

from utils import board_maker
from torch.utils.data import DataLoader

NUM_CONFIG = 1000

class GoL_Sup_Dataset:
    def __init__(self, board_dim=16, data_type='random', max_timestep=10, custom_features=False, split='Train'):
        self.data = []
        if os.path.exists('train_data_sup_'+data_type+'.data'):
            self.data = torch.load('train_data_sup_'+data_type+'.data')[:1000]
        else:
            if data_type == 'random':
                max_timestep = 1
                for _ in range(NUM_CONFIG):
                    distrib = torch.distributions.Bernoulli(0.05)
                    weights = torch.tensor([[1,1,1],[1,10,1],[1,1,1]]).view(1,1,3,3).float()
                    board = distrib.sample((board_dim, board_dim)).view(1,1,board_dim, board_dim)
                    board = board.to(torch.float32)
                    config_data = []
                    for _ in range(max_timestep):
                        newboard = F.conv2d(board, weights, padding=1).view(1,1,board_dim,board_dim)
                        newboard = (newboard==12) | (newboard==3) | (newboard==13)
                        newboard = newboard.to(torch.float32)
                        self.data += board.view(board_dim,board_dim), newboard.view(board_dim,board_dim)
                        board = newboard
                # breakpoint()
                self.data = torch.stack(self.data).reshape(NUM_CONFIG * max_timestep, 2, board_dim, board_dim)
                torch.save(self.data, 'train_data_sup_random.data')
            else:
                NUM_CONFIG = 100
                self.datacount = 0
                for filename in os.listdir('./all/'):
                    if filename.endswith(".rle"): 
                        #print(filename)
                        weights = torch.tensor([[1,1,1],[1,10,1],[1,1,1]]).view(1,1,3,3).float()
                        board = boardmaker.board_maker('./all/'+filename)
                        if board is None:
                            continue
                        board = board.view(1, 1, board_dim, board_dim)
                        self.datacount += 1
                        
                        #board = distrib.sample((board_dim, board_dim)).view(1,1,board_dim, board_dim)
                        board = board.to(torch.float32)
                        config_data = []
                        for _ in range(max_timestep):
                            newboard = F.conv2d(board, weights, padding=1).view(1,1,board_dim,board_dim)
                            newboard = (newboard==12) | (newboard==3) | (newboard==13)
                            newboard = newboard.to(torch.float32)
                            # newboard_array = np.int8(newboard) * 255
                            # img = Image.fromarray(newboard_array).convert('RGB')
                            # img = np.array(img)
                            # cv2.imshow("game", img)
                            # q = cv2.waitKey(100)
                            # if q == 113:
                            #     cv2.destroyAllWindows()
                            #     break
                            self.data += board.view(board_dim,board_dim), newboard.view(board_dim,board_dim)
                            board = newboard
                        if(self.datacount == NUM_CONFIG):
                            break
                self.data = torch.stack(self.data).reshape(self.datacount * max_timestep, 2, board_dim, board_dim)
                torch.save(self.data, 'train_data_sup_pattern.data')
        # breakpoint()

        self.data = self.data.float()
        self.data.requires_grad = False
        if custom_features == True:
            self.data = extract_custom_features(self.data)
            self.data.requires_grad = True
            torch.save(self.data, 'train_data_sup_'+data_type+'_'+'custom'+'.data')

        if split == 'Train':
            self.data = self.data[:int(len(self.data)*0.8)]
        elif split == 'Validation':
            self.data = self.data[int(len(self.data)*0.8):int(len(self.data)*0.9)]
        else:
            self.data = self.data[int(len(self.data)*0.9):]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

class OrderedGOLDataset:
    def __init__(self, data_dir='dataset_nodie/', data_type=None, data_order='mixed', board_dim=16, split='Train'):
        self.data = []
        if split == 'Validation':
            self.data = torch.load(data_dir + 'random_dev.data')
        elif split == 'Test':
            self.data = torch.load(data_dir + 'random_test.data')
        else:
            if data_order == 'mixed':
                self.data = torch.load(data_dir + 'mixed.data')
            else:
                assert data_type != None
                self.data = torch.load(data_dir + data_type + '.data')

        self.data = self.data.float()
        # self.data = self.data[:self.data.size()[0]]
        # if custom_features == True:
        #     self.data = extract_custom_features(self.data)
        #     self.data.requires_grad = True
        #     torch.save(self.data, 'train_data_sup_'+data_type+'_'+'custom'+'.data')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]


def extract_custom_features(board):
    weights = torch.FloatTensor([[[0,0,0],[1,1,1],[0,0,0]], [[0,1,0],[0,1,0],[0,1,0]], [[1,0,0],[0,1,0],[0,0,1]], [[0,0,1],[0,1,0],[1,0,0]],
    [[1,1,0],[1,0,0],[0,0,0]], [[0,1,1],[0,0,1],[0,0,0]], [[0,0,0],[1,0,0],[1,1,0]], [[0,0,0],[0,0,1],[0,1,1]]])
    n_filters, fw, fh = weights.size()
    weights = weights.view(n_filters, 1, fw, fh)

    n, _, w,h = board.size()
    board = board.to(torch.float32)
    board_reshaped = board.view(n*2, 1, w,h)

    filtered = F.conv2d(board_reshaped, weights, padding=1)
    avgpool = torch.nn.AvgPool2d(3)
    features = avgpool(filtered)
    features = features.reshape(n, 2, n_filters, w//3, h//3)

    return features

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
