import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, reduce, repeat


class MultiHeadDistanceLayer(pl.LightningModule):
    def __init__(self, num_head, head_dim, max_length, feature_dim):
        super(MultiHeadDistanceLayer, self).__init__()
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_length = max_length
        self.input_dim = feature_dim

        # query
        self.query = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=False)
        torch.nn.init.xavier_normal_(self.query.weight)
        # key
        self.key = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=False)
        torch.nn.init.xavier_normal_(self.key.weight)

        self.prior_mean = nn.parameter.Parameter((torch.rand((self.num_head, )) * 2 - 1), requires_grad=True)
        self.log_prior_std = nn.parameter.Parameter(torch.ones((self.num_head, )) * np.log(0.4), requires_grad=True)

        self.distances_matrix = torch.arange(self.max_length)[None, :] - torch.arange(self.max_length)[:, None]
        self.distances_matrix = torch.true_divide(self.distances_matrix, self.max_length)
        self.distances_matrix = self.distances_matrix

    def forward(self, inputs):
        # inputs (batch, channel, data_length)

        #query, key = inputs.transpose(1, 2), inputs.transpose(1, 2)
        query, key = rearrange(inputs, 'b c l -> b l c'), rearrange(inputs, 'b c l -> b l c')

        data_length = query.size(1)

        # (batch, data_length, head_dim * num_head)
        query = self.query(query)
        key = self.key(key)

        multi_head_query = rearrange(query, 'b l (nh hd) -> nh b l hd', hd=self.head_dim, nh=self.num_head)
        multi_head_key = rearrange(key, 'b l (nh hd) -> nh b l hd', hd=self.head_dim, nh=self.num_head)

        # prior (data_length, data_length)
        self.distances_matrix = self.distances_matrix.type_as(inputs)
        prior_array = repeat(self.distances_matrix, 'l1 l2 -> nh l1 l2', nh=self.num_head)
        prior_array = self.gaussian(prior_array, self.prior_mean[:, None, None], torch.exp(self.log_prior_std[:, None, None]))

        attention = torch.matmul(multi_head_query, multi_head_key.transpose(2, 3)) * (float(self.head_dim) ** -0.5)
        attention = attention * prior_array[:, None, :data_length, :data_length]
        attention = F.softmax(attention, dim=-1)

        distance = attention * self.distances_matrix[:data_length, :data_length]
        distance = reduce(distance, 'nh b l1 l2 -> nh b l1', 'sum')
        distance = rearrange(distance, 'nh b l -> b l nh')

        return distance, rearrange(attention, 'nh b l1 l2 -> b nh l1 l2')


    @staticmethod
    def gaussian(x, mean, std):
        return (1.0 / std / torch.sqrt(torch.tensor(2.0 * 3.1415926)))*torch.exp(-0.5 * (x - mean)**2.0 / std**2.0)