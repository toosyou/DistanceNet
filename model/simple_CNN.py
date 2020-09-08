import torch
import torch.nn as nn
import torch.nn.functional as F


class simple_CNN(nn.Module):
    def __init__(self, num_layers, in_ch, out_ch, input_length):
        super(simple_CNN, self).__init__()
        self.backbone = self.make_layer(num_layers, in_ch)

        self.ap = nn.AvgPool1d(kernel_size=6)

        #self.final1 = nn.Linear(in_ch*(2**num_layers), in_ch*(2**(num_layers-1)))
        #self.final2 = nn.Linear(in_ch*(2**(num_layers-1)), in_ch*(2**(num_layers-2)))
        self.final = nn.Linear(in_ch*(2**(num_layers)),1)

    def forward(self, x):
        y = self.backbone(x)
        y = self.ap(y)
        y = self.final(y.flatten(1))
        #y = self.final2(y)
        #y = self.final(y)
        return y

    def make_layer(self, num_layers, in_ch):
        seq = []
        for i in range(num_layers):
            seq.append(nn.Conv1d(in_ch*(2**i), in_ch*(2**(i+1)), kernel_size=3, padding=1))
            seq.append(nn.BatchNorm1d(in_ch*(2**(i+1))))
            seq.append(nn.MaxPool1d(kernel_size=2))
            seq.append(nn.ReLU())
        return nn.Sequential(*seq)