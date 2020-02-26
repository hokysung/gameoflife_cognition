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


def board_maker(filename):
    f=open(filename, "r")
    try:
        lines = f.readlines()
    except:
        return None
    encoded = ''
    j = 0
    for j in range(len(lines)):
        if lines[j][0] != '#':
            break
    for i in range(j+1, len(lines)):
        encoded += lines[i].split('\n')[0].split('!')[0]
    encoded += '$'

    a = lines[j].split(',')
    try:
        x = int(a[0][4:])
        y = int(a[1][5:])
    except:
        return None
    
    if x > 30 or y > 30:
        return None


    board = torch.zeros(x*y)

    decoded = decoder(encoded,x,y)
    if len(decoded) != x*y:
        #print("erroneous data file: ", filename)
        return None

    #print(decoded)
    #print(len(decoded))
    for i in range(x*y):
        if decoded[i] == 'o':
            board[i] = 1


    board = board.reshape(y,x)
    hor = 30 - x
    ver = 30 - y
    left = int(random.random() * hor)
    top = int(random.random() * ver)
    
    pad = nn.ConstantPad2d((left, hor -  left, top, ver-top), 0)
    
    board = pad(board)

    #print(board.shape)

    return board


#print(board_maker("./all/128p13.1.rle"))

