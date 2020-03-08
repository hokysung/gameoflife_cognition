from __future__ import print_function

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils

class BaselineCNN(nn.Module):
    def __init__(self, img_size=16, channels=1, kernel_size=3, n_filters=8):
        super(BaselineCNN, self).__init__()
        self.channels = channels
        self.n_filters = n_filters
        self.conv = nn.Sequential(
                        nn.Conv2d(channels, n_filters, kernel_size, padding=1),
                        nn.ReLU()
                    )
        self.proj = nn.Sequential(
                        nn.Linear(n_filters, 1),
                        nn.Sigmoid()
                    )

    def forward(self, pattern_input):
        batch_size = pattern_input[0]

        # breakpoint()
        pattern_input = pattern_input.unsqueeze(1)
        conv = self.conv(pattern_input)
        result = self.proj(conv.permute(0, 2, 3, 1))
        return result.squeeze(-1)

class CustomFeatureCNN(nn.Module):
    def __init__(self, img_size=16, channels=8, kernel_size=3, n_filters=8):
        super(CustomFeatureCNN, self).__init__()
        self.channels = channels
        self.n_filters = n_filters
        self.conv = nn.Sequential(
                        nn.Conv2d(channels, n_filters, kernel_size, padding=1),
                        nn.ReLU()
                    )
        self.proj = nn.Sequential(
                        nn.Linear(n_filters, 1),
                        nn.Sigmoid()
                    )

    def forward(self, pattern_input):
        batch_size = pattern_input[0]

        conv = self.conv(pattern_input)
        # result = self.proj(conv.permute(0, 2, 3, 1))
        return conv

class PatternEncoder(nn.Module):
    def __init__(self, img_size=16, channels=1, kernel_size=3, hidden_dim=256, n_filters=8, z_dim=128):
        super(PatternEncoder, self).__init__()
        self.n_filters = n_filters
        self.conv = nn.Sequential(
                        nn.Conv2d(channels, n_filters, kernel_size, padding=1),
                        nn.ReLU(),
                    )
        self.proj = nn.Sequential(
                        nn.Linear(n_filters * img_size**2, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, z_dim)
                        # nn.Sigmoid()
                    )
        self.img_size=img_size

    def forward(self, pattern_input):
        batch_size = pattern_input.shape[0]

        # breakpoint()
        conv = self.conv(pattern_input.unsqueeze(1))
        z = self.proj(conv.view(batch_size, -1))
        return z

class PatternDecoder(nn.Module):
    def __init__(self, channels=1, img_size=16, hidden_dim=256, z_dim=128, n_filters=8):
        super(PatternDecoder, self).__init__()
        self.conv = nn.Sequential(
                        nn.ConvTranspose2d(n_filters, n_filters, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(n_filters, channels, 1, padding=0))
        self.proj = nn.Sequential(
                        nn.Linear(z_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, n_filters * img_size**2),
                    )
        self.n_filters = n_filters
        self.channels = channels
        self.img_size = img_size

    def forward(self, z):
        if z.dim() != 1:
            batch_size = z.size(0)
        else:
            batch_size = 1

        # breakpoint()
        out = self.proj(z)
        out = out.view(batch_size, self.n_filters, self.img_size, self.img_size)
        out = self.conv(out)
        pattern_logits = out.view(batch_size, self.channels, self.img_size, self.img_size).squeeze(1)
        out_pattern = nn.Sigmoid()(pattern_logits)
        return out_pattern

# def gen_32_conv_output_dim(s):
#     s = get_conv_output_dim(s, 2, 0, 2)
#     s = get_conv_output_dim(s, 2, 0, 2)
#     s = get_conv_output_dim(s, 2, 0, 2)
#     return s

# def get_conv_output_dim(I, K, P, S):
#     # I = input height/length
#     # K = filter size
#     # P = padding
#     # S = stride
#     # O = output height/length
#     O = (I - K + 2*P)/float(S) + 1
#     return int(O)

# class ImageEncoder(nn.Module):
#     def __init__(self, channels, img_size, z_dim, n_filters=64):
#         super(ImageEncoder, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels, n_filters, 2, 2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0),
#             nn.ReLU())
#         cout = gen_32_conv_output_dim(img_size)
#         self.fc = nn.Linear(n_filters * 4 * cout**2, z_dim * 2)
#         self.cout = cout
#         self.n_filters = n_filters

#     def forward(self, img):
#         batch_size = img.size(0)
#         out = self.conv(img)
#         out = out.view(batch_size, self.n_filters * 4 * self.cout**2)
#         z_params = self.fc(out)
#         z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

#         return z_mu, z_logvar

# class ImageDecoder(nn.Module):
#     def __init__(self, channels, img_size, z_dim, n_filters=64):
#         super(ImageDecoder, self).__init__()
#         self.conv = nn.Sequential(
#             nn.ConvTranspose2d(n_filters * 4, n_filters * 4, 2, 2, padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, 2, padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose2d(n_filters * 2, n_filters, 2, 2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(n_filters, channels, 1, 1, padding=0))
#         cout = gen_32_conv_output_dim(img_size)
#         self.fc = nn.Sequential(
#             nn.Linear(z_dim, n_filters * 4 * cout**2),
#             nn.ReLU())
#         self.cout = cout
#         self.n_filters = n_filters
#         self.channels = channels
#         self.img_size = img_size

#     def forward(self, z):
#         if z.dim() != 1:
#             batch_size = z.size(0)
#         else:
#             batch_size = 1
#         out = self.fc(z)
#         out = out.view(batch_size, self.n_filters * 4, self.cout, self.cout)
#         out = self.conv(out)
#         x_logits = out.view(batch_size, self.channels, self.img_size, self.img_size)
#         x_mu = torch.sigmoid(x_logits)

#         return x_mu

# class ImageTextEncoder(nn.Module):
#     def __init__(self, channels, img_size, z_dim, embedding_module, text_hidden_dim=256, n_filters=64):
#         super(ImageTextEncoder, self).__init__()
#         self.text_embedding = embedding_module
#         self.embedding_dim = embedding_module.embedding.embedding_dim
#         self.text_model = nn.GRU(self.embedding_dim, text_hidden_dim)
#         self.image_model = nn.Sequential(
#             nn.Conv2d(channels, n_filters, 2, 2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0),
#             nn.ReLU())
#         cout = gen_32_conv_output_dim(img_size)
#         self.fc = nn.Linear(n_filters * 4 * cout**2 + text_hidden_dim, z_dim * 2)
#         self.cout = cout
#         self.n_filters = n_filters

#     def text_forward(self, seq, length):
#         batch_size = seq.size(0)

#         if batch_size > 1:
#             sorted_lengths, sorted_idx = torch.sort(length, descending=True)
#             seq = seq[sorted_idx]

#         # embed your sequences
#         embed_seq = self.text_embedding(seq)

#         # reorder from (B,L,D) to (L,B,D)
#         embed_seq = embed_seq.transpose(0, 1)

#         packed = rnn_utils.pack_padded_sequence(
#             embed_seq,
#             sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

#         _, hidden = self.text_model(packed)
#         hidden = hidden[-1, ...]

#         if batch_size > 1:
#             _, reversed_idx = torch.sort(sorted_idx)
#             hidden = hidden[reversed_idx]

#         return hidden

#     def forward(self, img, text, length):
#         batch_size = img.size(0)
#         out_img = self.image_model(img)
#         out_img = out_img.view(batch_size, self.n_filters * 4 * self.cout**2)
#         out_txt = self.text_forward(text, length)
#         out = torch.cat((out_img, out_txt), dim=1)
#         z_params = self.fc(out)
#         z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)

#         return z_mu, z_logvar
