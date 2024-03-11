import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

import numpy as np
import pywt
import time


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size=(13, 17)):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class FeedForwardNetwork(nn.Module):
    def __init__(self, freq_len, hidden_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(freq_len, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, 1)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.configs = configs
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)
        self.conv_r = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        self.fnn_r = FeedForwardNetwork(100, 2048)
        self.conv_i = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
        self.fnn_i = FeedForwardNetwork(100, 2048)
        self.merge = torch.nn.Conv2d(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            kernel_size=(2, 1)
        )

    def forward(self, seasonal_Embed):
        device = torch.device('cuda:0')
        tensor_3d = seasonal_Embed.permute(0, 2, 1)
        if str(tensor_3d.device) == 'cpu':
            ndarray_3d = tensor_3d.detach().numpy()
        else:
            ndarray_3d = tensor_3d.detach().cpu().numpy()
        wavelet = 'cgau8'
        totalscal = 100
        wfc = pywt.central_frequency(wavelet=wavelet)
        sample_rate = 1
        period = 1.0 / sample_rate
        a = 2 * wfc * totalscal / (np.arange(totalscal, 0, -1))
        [amp, f] = pywt.cwt(ndarray_3d, a, wavelet, period)
        tensor_4d_r = torch.tensor(np.real(amp).transpose((1, 2, 0, 3)), dtype=torch.float32).to(device)
        tensor_4d_i = torch.tensor(np.imag(amp).transpose((1, 2, 0, 3)), dtype=torch.float32).to(device)

        res_r = self.drop(self.act(self.conv_r(tensor_4d_r)))
        res_i = self.drop(self.act(self.conv_r(tensor_4d_i)))

        res_r = res_r.permute(0, 1, 3, 2)
        res_i = res_i.permute(0, 1, 3, 2)
        res_r = self.fnn_r(res_r)
        res_i = self.fnn_r(res_i)

        res_mg = torch.cat((res_r, res_i), dim=3)
        res_mg = res_mg.permute(0, 1, 3, 2)
        res_mg = self.merge(res_mg)
        res_mg = res_mg.squeeze(-2)
        res_mg = res_mg.permute(0, 2, 1)
        res_mg = res_mg[:, :(self.configs.seq_len + self.configs.pred_len), :]

        seasonal_out = res_mg + seasonal_Embed
        return seasonal_out


class Seasonal_Prediction(nn.Module):
    def __init__(self, configs):
        super(Seasonal_Prediction, self).__init__()

        self.configs = configs
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.model = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        self.task_name = configs.task_name
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(configs.seq_len, configs.pred_len + configs.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.projection = nn.Linear(configs.d_model, configs.d_model, bias=True)

        self.decomp_multi = series_decomp_multi()

    def forward(self, seasonal_Embed):
        for i in range(self.layer):
            seasonal_Embed, trend = self.decomp_multi(seasonal_Embed)
            seasonal_Embed = self.layer_norm(self.model[i](seasonal_Embed))
        seasonal_out = self.projection(seasonal_Embed)
        return seasonal_out


class Model(nn.Module):
    def __init__(self, configs, mode='regre'):
        super(Model, self).__init__()

        self.mode = mode
        self.decomp_multi = series_decomp_multi()
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )
        self.regression = nn.Linear(configs.seq_len, configs.pred_len)
        self.regression.weight = nn.Parameter((1/configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]), requires_grad=True)

        self.configs = configs
        self.task_name = configs.task_name
        self.conv_trans = Seasonal_Prediction(configs)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear((configs.d_model+3) * configs.seq_len, configs.num_class)

    def forecast(self, x, x_mark, y, y_mark):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        if self.mode == 'regre':
            seasonal, trend = self.decomp_multi(x)
            trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.mode == 'mean':
            mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.confis.pred_len, 1)
            seasonal, trend = self.decomp_multi(x)
            trend = torch.cat([trend[:, -self.configs.seq_len:, :], mean], dim=1)
        else:
            return None

        zeros = torch.zeros([x.shape[0], self.configs.pred_len, x.shape[2]], device=x.device)
        seasonal = torch.cat([seasonal[:, -self.configs.seq_len:, :], zeros], dim=1)
        seasonal_Embed = self.enc_embedding(seasonal, y_mark)

        seasonal_out = self.conv_trans(seasonal_Embed)
        y_pred = seasonal_out[:, -self.configs.pred_len:, :] + trend[:, -self.configs.pred_len:, :]

        y_pred = y_pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        y_pred = y_pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        return y_pred

    def imputation(self, x, x_mark, y, y_mark, mask):
        means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x /= stdev

        if self.mode == 'regre':
            seasonal, trend = self.decomp_multi(x)
            trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.mode == 'mean':
            mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.confis.pred_len, 1)
            seasonal, trend = self.decomp_multi(x)
            trend = torch.cat([trend[:, -self.configs.seq_len:, :], mean], dim=1)
        else:
            return None

        seasonal_Embed = self.enc_embedding(seasonal, x_mark)
        seasonal_out = self.conv_trans(seasonal_Embed)
        y_pred = seasonal_out + trend

        y_pred = y_pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        y_pred = y_pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return y_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None

