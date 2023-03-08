import os
import torch.nn as nn
import copy
from collections import deque
import random
import numpy as np
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, k_length):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('table', self.get_positional_encoding_table(model_dim, k_length))

    def get_positional_encoding_table(self, d_model, k):
        def get_pos_vector(pos):
            return [pos / np.power(10000, 2 * (i / d_model)) for i in range(d_model)]

        table = np.array([get_pos_vector(position) for position in range(k)])
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.sin(table[:, 1::2])

        return torch.FloatTensor(table).unsqueeze(0)


def ScaledDotProduct(Q, K, V, d):
   qkT = Q @ K.transpose(-2, 1)
   inner = qkT/math.sqrt(d)
   return torch.nn.Softmax(dim=-1)(inner) @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.linear1 = nn.Linear(emb, 3 * emb)
        self.emb = emb

    def forward(self, x):
        B, T, C = x.size()

        query, key, value = self.linear1(x).split(self.n_embd, dim=2)

        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        sdp = ScaledDotProduct(query, key, value, self.emb)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
