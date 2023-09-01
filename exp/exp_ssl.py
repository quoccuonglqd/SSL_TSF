from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, Ours, Byol
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from tsaug import Quantize, Drift, Reverse
my_augmenter = (
    Quantize(n_levels=[10, 20, 30]) + Drift(max_drift=(0.1, 0.5)) @ 0.8 + Reverse() @ 0.5
)

warnings.filterwarnings('ignore')

class Exp_SSL(Exp_Basic):
    def __init__(self, args):
        super(Exp_SSL, self).__init__(args)

    def _build_model(self):
        model_dict = {
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Informer': Informer,
            # 'DLinear': DLinear,
            # 'NLinear': NLinear,
            # 'Linear': Linear,
            'Ours': Ours
        }
        ssl_model_dict = {
            'BYOL': Byol.BYOL,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        ssl_model = ssl_model_dict[self.args.ssl_model_id](model, self.args.pred_len * self.args.enc_in, self.args.seq_len * 2, self.args.pred_len).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            ssl_model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return ssl_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = NegativeCosineSimilarity()
        return criterion 
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # time_now = time.time()

        train_steps = len(train_loader)
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            total_loss = 0
            momentum_val = cosine_schedule(epoch, self.args.train_epochs, 0.996, 1)
            for i, (batch_x, _, _, _) in enumerate(train_loader):
                augmented_batch_x = batch_x.clone()
                augmented_batch_x = augmented_batch_x.detach().cpu().numpy()
                augmented_batch_x = my_augmenter.augment(augmented_batch_x)
                augmented_batch_x = torch.from_numpy(augmented_batch_x)

                update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum_val)
                update_momentum(
                    self.model.projection_head, self.model.projection_head_momentum, m=momentum_val
                )

                batch_x = batch_x.float().to(self.device)
                augmented_batch_x = augmented_batch_x.float().to(self.device)

                p0 = self.model(batch_x)
                z0 = self.model.forward_momentum(batch_x)
                p1 = self.model(augmented_batch_x)
                z1 = self.model.forward_momentum(augmented_batch_x)

                loss = criterion(p0, z1) / 2 + criterion(p1, z0) / 2
                total_loss += loss.item()

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                if i % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i * len(batch_x), len(train_loader.dataset),
                        100. * i / len(train_loader),
                        loss.item() / len(batch_x)))
                    
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, total_loss / train_steps))
            
        model_path = os.path.join(path, 'checkpoint.pth')
        torch.save(self.model.backbone.state_dict(), model_path)

    

