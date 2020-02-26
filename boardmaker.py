import os
import torch
import torch.nn as nn
import random

#f=open("./all/4boats.rle", "r")

def decoder(data,x,y):
    decode = ''
    count = ''
    pos = 0
    
    for char in data:
        if char.isdigit():
            count += char
        elif char.isalpha():
            if count == '':
                num = 1
            else:
                num = int(count)
            decode += char * num
            count = ''
            pos+= num
            
        else:
            if count == '':
                num = 1
            else:
                num = int(count)
            decode += 'b' * (x - pos)
            decode += 'b' * x * (num-1)
            pos = 0
            count = ''
            
    return decode


def boardmaker(filename):
    f=open(filename, "r")
    lines = f.readlines()
    encoded = ''
    j = 0
    for j in range(len(lines)):
        if lines[j][0] != '#':
            break
    for i in range(j+1, len(lines)):
        encoded += lines[i].split('\n')[0].split('!')[0]
    encoded += '$'

    a = lines[j].split(',')
    x = int(a[0][4:])
    y = int(a[1][5:])
    
    if x > 50 or y > 50:
        return None


    board = torch.zeros(x*y)

    decoded = decoder(encoded,x,y)


    for i in range(x*y):
        if decoded[i] == 'o':
            board[i] = 1


    board = board.reshape(y,x)
    hor = 50 - x
    ver = 50 - y
    left = int(random.random() * hor)
    top = int(random.random() * ver)
    
    pad = nn.ConstantPad2d((left, hor -  left, top, ver-top), 0)
    
    board = pad(board)

    print(board.shape)

    return board


print(boardmaker("./all/2x2glider.rle"))

