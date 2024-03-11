import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # print("#################################### PE-Init-1 #####################################")
        super(PositionalEmbedding, self).__init__()

        d_model_raw = d_model
        if (d_model_raw % 2) != 0:
            d_model = d_model + 1

        # Compute the positional encodings once in log space.
        # print(max_len)                                              # 5000
        # print(d_model)                                              # 16
        pe = torch.zeros(max_len, d_model).float()
        # print(pe.shape)                                             # torch.Size([5000, 16])
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # print(position.shape)                                       # torch.Size([5000, 1])
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # print(div_term.shape)                                       # torch.Size([8])

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print((position * div_term).shape)                          # torch.Size([5000, 8])
        # print(pe.shape)                                             # torch.Size([5000, 16])
        pe = pe.unsqueeze(0)
        # print(pe.shape)                                             # torch.Size([1, 5000, 16])

        if (d_model_raw % 2) != 0:
            pe = pe[:, :, :d_model_raw]

        self.register_buffer('pe', pe)
        # print("#################################### PE-Init-2 #####################################")

    def forward(self, x):
        # print("#################################### PE-Forward-1 #####################################")
        # print(x.shape)                                              # torch.Size([32, 96, 7])
        # print(self.pe.shape)                                        # torch.Size([1, 5000, 16])
        output = self.pe[:, :x.shape[1]]
        # print(output.shape)                                         # torch.Size([1, 96, 16])
        # print("#################################### PE-Forward-2 #####################################")
        return output


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print("#################################### TokenEmbedding-1 #####################################")
        # print(x.shape)                                          # torch.Size([32, 96, 7])   torch.Size([224, 12, 16])
        # print(x.permute(0, 2, 1).shape)                         # torch.Size([32, 7, 96])   torch.Size([224, 16, 12])
        # print(self.tokenConv(x.permute(0, 2, 1)).shape)         # torch.Size([32, 16, 96])  torch.Size([224, 16, 12])
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # print(x.shape)                                          # torch.Size([32, 96, 16])  torch.Size([224, 12, 16])
        # print("#################################### TokenEmbedding-2 #####################################")
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print("######################### PatchEmbedding-1 ########################")
        # print(x.shape)                      # torch.Size([32, 7, 96])
        n_channel = x.shape[1]
        x = self.padding_patch_layer(x)
        # print(x.shape)                      # torch.Size([32, 7, 104])
        # print(self.patch_len)               # 16
        # print(self.stride)                  # 8
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # print(x.shape)                      # torch.Size([32, 7, 12, 16])
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # print(x.shape)                      # torch.Size([224, 12, 16])

        # 2、EncoderEmbedding
        ve = self.value_embedding(x)
        # print(ve.shape)                     # torch.Size([224, 12, 16])
        pe = self.position_embedding(x)
        # print(pe.shape)                     # torch.Size([1, 12, 16])
        x_embed = ve + pe
        # print(x_embed.shape)                # torch.Size([224, 12, 16])
        # print(n_vars)                       # 7
        x_embed = self.dropout(x_embed)
        # print("######################### PatchEmbedding-2 ########################")
        return x_embed, n_channel


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        # print("#################################### TemporalEmbedding-1 #####################################")
        # print(x.shape)
        x = x.long()
        # print(x[:, :, 4].shape)
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        # print(minute_x.shape)
        # print(x[:, :, 3].shape)
        hour_x = self.hour_embed(x[:, :, 3])
        # print(hour_x.shape)
        # print(x[:, :, 2].shape)
        weekday_x = self.weekday_embed(x[:, :, 2])
        # print(weekday_x.shape)
        # print(x[:, :, 1].shape)
        day_x = self.day_embed(x[:, :, 1])
        # print(day_x.shape)
        # print(x[:, :, 0].shape)
        month_x = self.month_embed(x[:, :, 0])
        # print(month_x.shape)
        output = hour_x + weekday_x + day_x + month_x + minute_x
        # print(output.shape)
        # print("#################################### TemporalEmbedding-2 #####################################")
        return output


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        # print("#################################### TimeFeatureEmbedding-1 #####################################")
        # print(x.shape)                      # torch.Size([32, 192, 4])
        output = self.embed(x)
        # print(output.shape)                 # torch.Size([32, 192, 16])
        # print("#################################### TimeFeatureEmbedding-2 #####################################")
        return output


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = \
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else \
            TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # print("#################################### DataEmbedding-1 #####################################")
        if x_mark is None:
            # print("x_mark is None")
            # print(x.shape)                                      # torch.Size([16, 36, 1])
            # print(self.value_embedding(x).shape)                # torch.Size([16, 36, 32])
            # print(self.position_embedding(x).shape)             # torch.Size([1, 36, 32])
            x = self.value_embedding(x) + self.position_embedding(x)
            # print(x.shape)                                      # torch.Size([16, 36, 32])
        else:
            # print("x_mark is not None")
            # print(x.shape)                                      # torch.Size([32, 192, 7])
            # print(x_mark.shape)                                 # torch.Size([32, 192, 4])
            # print(self.value_embedding(x).shape)                # torch.Size([32, 192, 16])
            # print(self.position_embedding(x).shape)             # torch.Size([1, 192, 16])
            # print(self.temporal_embedding(x_mark).shape)        # torch.Size([32, 192, 16])
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
            # print(x.shape)                                      # torch.Size([32, 192, 16])
        # print("#################################### DataEmbedding-2 #####################################")
        return self.dropout(x)