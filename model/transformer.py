import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_positional_encoding(input_shape, temperature=10000):
    signal_length, feature_dim = input_shape
    embed = torch.range(0, signal_length-1, dtype=torch.float32) # (signal_length)

    dim_t = torch.range(0, feature_dim-1, dtype=torch.float32) # (feature_dim)
    dim_t = temperature ** (2 * (dim_t // 2) / feature_dim)

    pos = embed.unsqueeze(-1) / dim_t # (signal_length, feature_dim)
    pos = torch.stack([torch.sin(pos[...,0::2]), torch.cos(pos[..., 1::2])], axis=2)

    pos = pos.view((1, signal_length, -1))

    return pos # (1, signal_length, feature_dim)

class MultiHeadDistanceLayer(nn.Module):
    def __init__(self, num_head, head_dim, max_length, feature_dim):
        super(MultiHeadDistanceLayer, self).__init__()
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_length = max_length
        self.input_dim = feature_dim

        # query
        self.query = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=True)
        torch.nn.init.xavier_normal_(self.query.weight)
        # key
        self.key = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=True)
        torch.nn.init.xavier_normal_(self.key.weight)
        # value
        self.value = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=True)
        torch.nn.init.xavier_normal_(self.value.weight)


        self.prior_mean = nn.parameter.Parameter(torch.rand((self.num_head, )) * self.max_length * 2 - self.max_length, requires_grad=True)
        self.log_prior_std = nn.parameter.Parameter(torch.ones((self.num_head, )) * np.log(self.max_length / 4), requires_grad=True)

        self.distances_matrix = torch.arange(self.max_length)[None, :] - torch.arange(self.max_length)[:, None]
        self.distances_matrix = self.distances_matrix.cuda()

        self.position_embeding = get_positional_encoding((max_length, feature_dim)).cuda()

    def forward(self, inputs, return_attention=True):

        query = key = inputs.transpose(1, 2) + self.position_embeding

        value = inputs.transpose(1, 2)

        data_length = query.size(1)

        # embedding
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        multi_head_query = torch.cat(torch.split(query, self.head_dim, dim=2), dim=0)
        multi_head_key = torch.cat(torch.split(key, self.head_dim, dim=2), dim=0)
        multi_head_value = torch.cat(torch.split(key, self.head_dim, dim=2), dim=0)

        attention = torch.matmul(multi_head_query, multi_head_key.transpose(1, 2)) * (float(self.head_dim) ** -0.5)
        attention = F.softmax(attention, dim=-1)

        distance = torch.matmul(attention, multi_head_value)

        distance = torch.cat(torch.split(distance, distance.size(0) // self.num_head, dim=0), dim=-1)

        if return_attention:
            return distance, torch.cat(torch.split(attention[..., None], distance.size(0) // self.num_head, dim=0), dim=-1)
        return distance


    @staticmethod
    def gaussian(x, mean, std):
        return (1.0 / std / torch.sqrt(torch.tensor(2.0 * 3.1415926)))*torch.exp(-0.5 * (x - mean)**2.0 / std**2.0)