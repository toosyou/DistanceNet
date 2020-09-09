import torch
import torch.nn as nn
import torch.nn.functional as F


class simple_CNN(nn.Module):
    def __init__(self, num_layers, in_ch, out_ch, input_length):
        super(simple_CNN, self).__init__()
        self.base = 16
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool1d(kernel_size=2)
        self.backbone = self.make_layer(num_layers, in_ch, base=self.base)

        self.ap = nn.AvgPool1d(kernel_size=input_length // (2**num_layers))
        self.final = nn.Linear(self.base*in_ch*(2**(num_layers)),1)

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        y = self.backbone(x)
        y = self.ap(y)
        y = self.final(y.flatten(1))
        return y

    def make_layer(self, num_layers, in_ch, base=16):
        seq = []
        for i in range(num_layers):
            seq.append(nn.Conv1d(base*in_ch*(2**i), base*in_ch*(2**(i+1)), kernel_size=3, padding=1))
            seq.append(nn.ReLU())
            if i != num_layers-1:
                #seq.append(nn.BatchNorm1d(base*in_ch*(2**(i+1))))
                seq.append(nn.MaxPool1d(kernel_size=2))
        return nn.Sequential(*seq)