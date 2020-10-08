import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadDistanceLayer(nn.Module):
    def __init__(self, num_head, head_dim, max_length, feature_dim):
        super(MultiHeadDistanceLayer, self).__init__()
        self.num_head = num_head
        self.head_dim = head_dim
        self.max_length = max_length
        self.input_dim = feature_dim

        self.query = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=True)
        torch.nn.init.xavier_normal_(self.query.weight)
        #torch.nn.init.kaiming_normal_(self.query.weight)
        self.key = nn.Linear(self.input_dim, self.num_head*self.head_dim, bias=True)
        torch.nn.init.xavier_normal_(self.key.weight)
        #torch.nn.init.kaiming_normal_(self.key.weight)

        self.prior_mean = torch.zeros((1, ), requires_grad=True).cuda()
        self.prior_std = torch.ones((1, ), requires_grad=True).cuda()

        self.distances_matrix = torch.arange(self.max_length)[None, :] - torch.arange(self.max_length)[:, None]
        self.distances_matrix = self.distances_matrix.cuda()

    def forward(self, inputs):
        query, key = inputs.transpose(1, 2), inputs.transpose(1, 2)

        data_length = query.size(1)

        query = self.query(query)
        key = self.key(key)

        multi_head_query = torch.cat(torch.split(query, self.head_dim, dim=2), dim=0)
        multi_head_key = torch.cat(torch.split(key, self.head_dim, dim=2), dim=0)

        prior_array = self.gaussian(self.distances_matrix, self.prior_mean, self.prior_std)

        attension = torch.matmul(multi_head_query, multi_head_key.transpose(1, 2)) * (float(self.head_dim) ** -0.5)
        attension = attension * prior_array[:data_length, :data_length]
        attension = F.softmax(attension, dim=-1)

        distance = attension * self.distances_matrix[:data_length, :data_length]
        distance = torch.sum(distance, -1)

        distance = distance.unsqueeze(-1)
        distance = torch.cat(torch.split(distance, distance.size(0) // self.num_head, dim=0), dim=-1)

        return distance

    @staticmethod
    def gaussian(x, mean, std):
        return (1.0 / std / torch.sqrt(torch.tensor(2.0 * 3.1415926)))*torch.exp(-0.5 * (x - mean)**2.0 / std**2.0)
