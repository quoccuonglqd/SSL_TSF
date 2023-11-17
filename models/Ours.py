import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

from models.embed import *
from models.TSMixer import ResBlock

import pdb

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class ResBlock(nn.Module):
    def __init__(self, norm_type, dropout, ff_dim, seq_len, channel_dim, out_len):
        super(ResBlock, self).__init__()
        
        if norm_type == 'L':
            self.norm = nn.LayerNorm
        else:
            self.norm = nn.BatchNorm2d
        
        self.temporal_linear = nn.Sequential(
            self.norm(channel_dim),
            nn.ReLU(),
            nn.Linear(seq_len, out_len),
            nn.Dropout(dropout)
        )
        
        self.feature_linear = nn.Sequential(
            self.norm(seq_len),
            nn.ReLU(),
            nn.Linear(channel_dim, ff_dim),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, channel_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        res = x.clone()                                                  # (batch_size, seq_len, channel_dim)                                                
        x = self.temporal_linear(x) + res  

        res = x.clone()
        x = rearrange(self.feature_linear(rearrange(x, 'b c p l -> b l p c')), 'b l p c -> b c p l') + res
        
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, input_len, output_len):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.query = nn.Linear(d_model, d_model * n_head)
        self.key = nn.Linear(d_model, d_model * n_head)
        self.value = nn.Linear(d_model, d_model * n_head)

        self.linear = nn.Linear(input_len, output_len)

    def forward(self, query, key, value):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = rearrange(query, 'b l (h d) -> b h l d', h=self.n_head)
        key = rearrange(key, 'b l (h d) -> b h l d', h=self.n_head)
        value = rearrange(value, 'b l (h d) -> b h l d', h=self.n_head)

        attn = torch.einsum('bhld,bhkd->bhkl', query, key) / np.sqrt(self.d_model)
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum('bhkl,bhld->bhkd', attn, value)
        out = reduce(out, 'b h l d -> b l d', 'mean')
        out = self.linear(rearrange(out, 'b l d -> b d l'))
        return out


                
class MLPMixer(nn.Module):
    def __init__(self, seq_len, pred_len, channels, patch_size, dim, depth, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.5):
        super(MLPMixer, self).__init__()
        assert (seq_len % patch_size) == 0, 'seq length must be divisible by patch size'
        num_patches = (seq_len // patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.depth = depth

        for _ in range(self.depth):
            setattr(self, f'PreNormResidual_{_}', PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last)))
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(patch_size * channels, dim)
        self.linear2 = nn.Linear(dim, patch_size * channels)
        self.linear3 = nn.Linear(seq_len, seq_len + pred_len)
        self.rearrange1 = Rearrange('b c (l p) -> b l (p c)', p = patch_size)
        self.rearrange2 = Rearrange('b l (p c) -> b c (l p)', p = patch_size)
    
    def forward(self, x):
        import pdb; 
        x = self.rearrange1(x)
        x = self.linear1(x)
        for _ in range(self.depth):
            x = getattr(self, f'PreNormResidual_{_}')(x)
        
        x = self.norm(x)
        x = self.linear2(x)
        x = self.rearrange2(x)
        x = self.linear3(x)
        return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
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
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scales = configs.scales
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.dim = configs.d_model
        self.depth = configs.depth
        self.ff_dim = configs.ff_dim
        self.dropout = configs.dropout
        self.norm_type = configs.norm_type

        self.inverse_scale = 1 / np.array(self.scales, dtype=np.float32)
        device = 'cuda' if configs.use_gpu and torch.cuda.is_available() else 'cpu'
        self.scale_weight = F.softmax(torch.from_numpy(self.inverse_scale).to(device), dim=0)
        kernel_size = [17,49]
        self.decompsition = series_decomp(kernel_size)
        self.init_layer(None)

    def init_layer(self, sample_data):
        # self.revin_layer = RevIN(self.channels, affine=True, subtract_last=True)

        self.seasonal_backbones = nn.ModuleList()
        # self.pooling_layers = nn.ModuleList()
        for scale in self.scales:
            ceiled_seq_len = np.ceil(self.seq_len / scale).astype(int)
            ceiled_pred_len = np.ceil(self.pred_len / scale).astype(int)

            if self.individual:
                mlp_layer = nn.ModuleDict() 
                for i in range(self.channels):
                    mlp_layer[str(i)] = nn.Linear(ceiled_seq_len  + ceiled_pred_len , ceiled_seq_len + ceiled_pred_len)

                self.seasonal_backbones.append(mlp_layer)
            else:
                self.seasonal_backbones.append(ResBlock(self.norm_type, self.dropout, self.ff_dim, ceiled_seq_len + ceiled_pred_len, self.channels, ceiled_seq_len + ceiled_pred_len))
                # self.seasonal_backbones.append(CrossAttention(self.channels, 2, self.dropout, ceiled_seq_len + ceiled_pred_len, ceiled_seq_len + ceiled_pred_len))

            # self.pooling_layers.append(nn.MaxPool1d(kernel_size=scale, stride=scale, padding=0))


        # if self.individual:
        #     self.trend_head = nn.ModuleDict()
        #     # self.seasonal_head = nn.ModuleDict()
        #     for i in range(self.channels):
        #         self.trend_head[str(i)] = nn.Linear(self.seq_len, self.pred_len)
        #         # self.seasonal_head[str(i)] = nn.Linear(self.pred_len, self.pred_len)
        # else:
        self.trend_head = nn.Linear(self.seq_len, self.pred_len)
        self.seasonal_head = nn.Linear(self.pred_len, self.pred_len)

        self.embedding = PositionalEmbedding(self.channels)
            

    def exponential_smoothing(self, series, alpha):
        result = series[0]
        for value in series[1:]:
            result = alpha * value + (1 - alpha) * result
        return result
    
    def find_period(self, series, k = 2):
        xf = torch.fft.rfft(series, dim = 1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = series.shape[1] // top_list
        return period, abs(series).mean(-1)[:, top_list]

    def forward_backbone(self, x):
        result = []
        # x: [Batch, Channel, Input length]
        
        _, seasonality = self.decompsition(x)
        
        x = seasonality
        # x = self.padding_patch_layer(x)
        
        mask_pred = torch.zeros(x.shape[0], x.shape[1], self.pred_len).to(x.device)
        x = torch.cat([x, mask_pred], dim=2)

        x = x + self.embedding(x.permute(0,2,1)).permute(0,2,1)

        for ind in range(len(self.scales)):
            prev_x = x.clone()
            if ind == 0:
                mean = x[:, :, :self.seq_len].mean(-1, keepdim=True)
                x = x - mean
            else:
                mean = x.mean(-1, keepdim=True)
                x = x - mean

            if self.individual:
                z = torch.stack([self.seasonal_backbones[ind][str(i)](x[:,i,:]
                        .unfold(-1, self.scales[ind], self.scales[ind]).permute(0,2,1)) 
                        for i in range(self.channels)], dim=1)
            else:
                z = self.seasonal_backbones[ind](rearrange(x.unfold(-1, self.scales[ind], self.scales[ind]), 'b c l p -> b c p l'))
            
            #     z = rearrange(self.pooling_layers[ind](x), 'b c l -> b l c')
            #     positional_embedding = self.embedding(z)
            #     z = self.seasonal_backbones[ind](positional_embedding, z, z)


            # z = F.interpolate(z, size=self.seq_len + self.pred_len, mode='linear')
            z = nn.Flatten(2,3)(z)

            x = prev_x[:,:,:self.seq_len ] - z[:,:,:self.seq_len]
            z = z[:,:,self.seq_len:]

            # Concat the predicted z with the remaining x
            x = torch.cat([x, z], dim=2)

            result.append(z + mean)

        # result: [no.scales, [Batch, Output length, Channel]]
        result = torch.stack(result, dim=0)
        result = result.mean(0)

        result = torch.sum(result * self.scale_weight.view(-1,1,1,1), dim=0)
        return result
    
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)
        
        # x = self.revin_layer(x, mode='norm')
        
        result = self.forward_backbone(x)

        trend, _ = self.decompsition(x)
        
        # if self.individual:
        #     trend = torch.stack([self.trend_head[str(i)](trend[:,i,:]) for i in range(self.channels)], dim=1)
        #     result = self.seasonal_head(result)
        # else:
        trend = self.trend_head(trend)
        result = self.seasonal_head(result)
        
        result = result + trend

        # result = self.revin_layer(result.permute(0,2,1), mode='denorm')

        return result.permute(0,2,1)