import numpy as np
import sys
import torch
import torch.nn.functional as F

import numpy as np


def convolve(board):
    weights = torch.tensor([[[0,0,0],[1,1,1],[0,0,0]], [[0,1,0],[0,1,0],[0,1,0]], [[1,0,0],[0,1,0],[0,0,1]], [[0,0,1],[0,1,0],[1,0,0]],
    [[1,1,0],[1,0,0],[0,0,0]], [[0,1,1],[0,0,1],[0,0,0]], [[0,0,0],[1,0,0],[1,1,0]], [[0,0,0],[0,0,1],[0,1,1]]])

    w,h = board.size()
    board = board.view(1,1, w,h).to(torch.float32)
    filtered = None
    temp = []
    for we in weights:
        we = we.view(1,1,3,3).float()
        f = F.conv2d(board, we, padding=1).view(1,1,w,h)
        temp.append(f)

    filtered = torch.stack(temp, dim=0)
    return filtered.squeeze()


board = torch.tensor([[1,0,1],[0,1,0],[1,0,1]])

#print(convolve(board))
print(convolve(board))