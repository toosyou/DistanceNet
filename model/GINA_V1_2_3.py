import numpy as np
from numpy.core.numeric import zeros_like
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, reduce, repeat


class MultiHeadDistanceLayer(pl.LightningModule):
    def __init__(self, num_head, head_dim, max_length, feature_dim, window=3, mode='global'):
        super(MultiHeadDistanceLayer, self).__init__()
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_length = max_length
        self.input_dim = feature_dim
        self.mode = mode
        self.window = window

        # query
        self.query = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=False)
        torch.nn.init.xavier_normal_(self.query.weight)
        # key
        self.key = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=False)
        torch.nn.init.xavier_normal_(self.key.weight)
        # value
        self.value = nn.Linear(self.input_dim, self.num_head, bias=False)
        torch.nn.init.xavier_normal_(self.value.weight)

        self.prior_mean = nn.parameter.Parameter((torch.rand((self.num_head, )) * 2 - 1), requires_grad=True)
        self.log_prior_std = nn.parameter.Parameter(torch.ones((self.num_head, )) * np.log(0.4), requires_grad=True)

        self.distances_matrix = torch.arange(self.max_length)[None, :] - torch.arange(self.max_length)[:, None]
        self.distances_matrix = torch.true_divide(self.distances_matrix, self.max_length)

        self.learned_pe = nn.Parameter(torch.randn(1, self.max_length, self.input_dim))

    def forward(self, inputs):
        # inputs (batch, channel, data_length)

        #query, key = inputs.transpose(1, 2), inputs.transpose(1, 2)
        query, key = rearrange(inputs, 'b c l -> b l c') + self.learned_pe, rearrange(inputs, 'b c l -> b l c') + self.learned_pe
        value = rearrange(inputs, 'b c l -> b l c')

        data_length = query.size(1)

        # (batch, data_length, head_dim * num_head)
        query = self.query(query)
        key = self.key(key)
        value = torch.sigmoid(self.value(value))

        multi_head_query = rearrange(query, 'b l (nh hd) -> nh b l hd', hd=self.head_dim, nh=self.num_head)
        multi_head_key = rearrange(key, 'b l (nh hd) -> nh b l hd', hd=self.head_dim, nh=self.num_head)
        multi_head_value = rearrange(value, 'b l nh -> nh b l')

        # prior (data_length, data_length)
        self.distances_matrix = self.distances_matrix.type_as(inputs)
        prior_array = repeat(self.distances_matrix, 'l1 l2 -> nh l1 l2', nh=self.num_head)
        prior_array = self.gaussian(prior_array, self.prior_mean[:, None, None], torch.exp(self.log_prior_std[:, None, None]))

        attention = torch.matmul(multi_head_query, multi_head_key.transpose(2, 3)) * (float(self.head_dim) ** -0.5)
        attention = attention * prior_array[:, None, :data_length, :data_length]
        attention = F.softmax(attention, dim=-1)
        # (nh, b, l1, l2)

        attention = torch.tril(attention)

        attention = attention * multi_head_value.unsqueeze(-1) # (nh, b, l1, l2) * (nh, b, l, 1) = (nh, b, l1, l2)

        attention = F.avg_pool2d(attention, (1, self.window), stride=(1, 1), padding=(0, self.window//2))

        # attention with size [nh, b, l, l*2]
        attention = rearrange(attention, 'nh b l1 l2 -> b l1 l2 nh')
        

        if self.mode == 'global':
            attention = torch.sum(attention, dim=1) # (b, l2, nh)

        return attention, attention


    @staticmethod
    def gaussian(x, mean, std):
        return (1.0 / std / torch.sqrt(torch.tensor(2.0 * 3.1415926)))*torch.exp(-0.5 * (x - mean)**2.0 / std**2.0)