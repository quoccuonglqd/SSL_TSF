import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
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
        self.inverse_scale = 1 / np.array(self.scales, dtype=np.float32)
        device = 'cuda' if configs.use_gpu and torch.cuda.is_available() else 'cpu'
        self.scale_weight = F.softmax(torch.from_numpy(self.inverse_scale).to(device), dim=0)
        kernel_size = [7,13,25]
        self.decompsition = series_decomp(kernel_size)
        self.init_layer(None)

    def init_layer(self, sample_data):
        self.mlp_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        for scale in self.scales:
            ceiled_seq_len = np.ceil(self.seq_len / scale).astype(int)
            ceiled_pred_len = np.ceil(self.pred_len / scale).astype(int)
            self.mlp_layers.append(nn.Sequential(*[
                                   nn.Linear(ceiled_seq_len, ceiled_seq_len * 2),  
                                   nn.Linear(ceiled_seq_len * 2, ceiled_seq_len + ceiled_pred_len)
                                   ])
                                   )
            self.pooling_layers.append(nn.AvgPool1d(kernel_size=scale, stride=scale, padding=0, ceil_mode=True))

        self.trend_mlp = nn.Linear(self.seq_len, self.pred_len)
            

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

    def forward(self, x):
        result = []
        trend, seasonality = self.decompsition(x)
        # x: [Batch, Input length, Channel]
        # x = x.permute(0,2,1)
        x = seasonality.permute(0,2,1)
        for ind in range(len(self.scales)):
            prev_x = x.clone()
            x = self.pooling_layers[ind](x)
            # seq_last = x[:,:,-1:]
            # x = x - seq_last
            z = self.mlp_layers[ind](x)

            # z = z + seq_last
            z = F.interpolate(z, size=self.seq_len + self.pred_len, mode='linear')

            x = prev_x - z[:,:,:self.seq_len]
            z = z[:,:,self.seq_len:]
            result.append(z.permute(0,2,1))

        # result: [no.scales, [Batch, Output length, Channel]]
        result = torch.stack(result, dim=0)
        result = result.mean(0)

        trend = trend.permute(0,2,1)
        trend = self.trend_mlp(trend)
        trend = trend.permute(0,2,1)

        result = torch.sum(result * self.scale_weight.view(-1,1,1,1), dim=0)
        result = result + trend

        return result 