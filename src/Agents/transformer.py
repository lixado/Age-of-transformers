import torch.nn as nn
import numpy as np
import torch
import math

class AddAndNorm(nn.Module):
    def __init__(self):
        super().__init__()

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


def ScaledDotProduct(Q, K, V, d, mask=None):
   qkT = Q @ K.transpose(-2, 1)
   inner = qkT/math.sqrt(d)

   if mask is not None:
        inner = inner.masked_fill(mask==0, -1e9)

   return torch.nn.Softmax(dim=-1)(inner) @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 3 * d_model)
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        query, key, value = self.linear1(x).split(self.n_embd, dim=2)

        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        sdp = ScaledDotProduct(query, key, value, self.d_model, mask=mask)

        x += sdp # Add
        x = self.layer_norm(x) # Norm

        return x

class FeedForward(nn.Module):
    def __init__(self, d_inner, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_inner, d_model)
        self.w2 = nn.Linear(d_inner, d_model)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_inner, eps=1e-6)

    def forward(self, x):
        input = self.w1(x)
        input = self.activation(input)
        input = self.w2(input)

        x += input # Add
        x = self.layer_norm(x) #Norm

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner):
        super(EncoderLayer).__init__()
        self.attention = MultiHeadAttention(d_model)
        self.feed_forward = FeedForward(d_inner, d_model)

    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, d_inner):
        super().__init__()

        self.src_state_emb = nn.Embedding()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner) for _ in range(n_layers) # *N
        ])

    def forward(self):
        pass

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner):
        super(DecoderLayer).__init__()
        self.attention = MultiHeadAttention(d_model)
        self.encoded_attention = MultiHeadAttention(d_model)
        self.feed_forward = FeedForward(d_inner, d_model)

    def forward(self, input, output, mask=None):
        output = self.attention(input, mask=mask)
        output = self.encoded_attention(output)
        output = self.feed_forward(output)
        return output


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, d_inner):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner) for _ in range(n_layers)  # *N
        ])

    def forward(self):
        pass


class Transformer(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
