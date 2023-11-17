import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, norm_type, dropout, ff_dim, seq_len, channel_dim):
        super(ResBlock, self).__init__()
        
        if norm_type == 'L':
            self.norm = nn.LayerNorm
        else:
            self.norm = nn.BatchNorm1d
        
        self.temporal_linear = nn.Sequential(
            self.norm(channel_dim),
            nn.ReLU(),
            nn.Linear(seq_len, seq_len),
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
        res = x.clone()                                                    # (batch_size, seq_len, channel_dim)
        x = self.temporal_linear(x.transpose(1, 2)).transpose(1, 2) + res  # (batch_size, seq_len, channel_dim)

        res = x.clone()
        x = self.feature_linear(x) + res
        
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.blocks = nn.ModuleList([ResBlock(configs.norm_type, configs.dropout, configs.ff_dim, configs.seq_len, configs.enc_in) for _ in range(configs.n_block)])
        
        self.target_slice = configs.target_slice
        
        self.temporal_linear = nn.Sequential(
            nn.Linear(configs.seq_len, configs.pred_len)
        )
    
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        for block in self.blocks:
            x = block(x)
        
        if self.target_slice:
            x = x[:, :, self.target_slice]
        
        x = self.temporal_linear(x.transpose(1, 2)).transpose(1, 2)
        
        return x