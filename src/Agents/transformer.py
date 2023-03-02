import os
import torch.nn as nn
import copy
from collections import deque
import random
import numpy as np
import torch
import math

def ScaledDotProduct(Q, K, V, d):
   qkT = Q @ K.transpose(-2, 1)
   inner = qkT/math.sqrt(d)
   return nn.Softmax(inner)*V

class MultiHeadAttention(nn.Module):
    def __init__(self, emb):
        self.linear1 = nn.Linear(emb, 3 * emb)
        self.scaled_dot_product =

    def forward(self, x):
        B, T, C = x.size()

        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)