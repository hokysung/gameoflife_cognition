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

from torch.utils.data import DataLoader

NUM_CONFIG = 200

class GoL_Sup_Dataset:
    def __init__(self, board_dim=30, types='random', max_timestep=10, split='Train'):
        self.data = []
        # if os.path.exists('train_data_sup.data'):
        #     self.data = torch.load('train_data_sup.data')
        #     return
        # else:
        #     self.data = []

        if types == 'random':
            for _ in range(NUM_CONFIG):
                distrib = torch.distributions.Bernoulli(0.5)
                weights = torch.tensor([[1,1,1],[1,10,1],[1,1,1]]).view(1,1,3,3).float()
                board = distrib.sample((board_dim, board_dim)).view(1,1,board_dim, board_dim)
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
            # breakpoint()
            self.data = torch.stack(self.data).reshape(NUM_CONFIG * max_timestep, 2, board_dim, board_dim)
            torch.save(self.data, 'train_data_sup.data')
        else:
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
                    if(self.datacount == 200):
                        break

            # breakpoint()

            self.data = torch.stack(self.data).reshape(self.datacount * max_timestep, 2, board_dim, board_dim)

            self.data = self.data[:50]
            torch.save(self.data, 'train_data_sup_pattern.data')

        self.data = self.data.float()
        self.data.requires_grad = True

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

# sup_dataset = GoL_Sup_Dataset()
# sup_dataloader = DataLoader(sup_dataset, shuffle=True, batch_size=5)
# for batch_idx, (prev_board, next_board) in enumerate(sup_dataloader):
#     print(batch_idx)
#     print(prev_board)
#     print(next_board)

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

# BOARD_HEIGHT = 100
# BOARD_WIDTH = 100

# distrib = torch.distributions.Bernoulli(0.7)

# weights = torch.tensor([[1,1,1],[1,10,1],[1,1,1]]).view(1,1,3,3)
# board = distrib.sample((BOARD_HEIGHT,BOARD_WIDTH)).view(1,1,BOARD_HEIGHT,BOARD_WIDTH)
# board = board.to(torch.int64)

# cv2.namedWindow("game", cv2.WINDOW_NORMAL)

# if False:
#     data = torch.load('train.data')
#     data.append(board)
# else:
#     data = [board]

# while True:
#     newboard = F.conv2d(board, weights, padding=1).view(BOARD_HEIGHT,BOARD_WIDTH)
#     newboard = (newboard==12) | (newboard==3) | (newboard==13)
#     newboard_array = np.int8(newboard) * 255
#     img = Image.fromarray(newboard_array).convert('RGB')
#     img = np.array(img)
#     cv2.imshow("game", img)
#     q = cv2.waitKey(100)
#     if q == 113: # 'q'
#         cv2.destroyAllWindows()
#         break
#     board = torch.tensor(newboard_array/255, dtype=torch.int64).view(1,1,BOARD_HEIGHT,BOARD_WIDTH)
#     data.append(board)
# torch.save(data, 'train.data')

#dataset = GoL_Sup_Dataset(types = 'not_random')