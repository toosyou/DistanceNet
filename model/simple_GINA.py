import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.distance import MultiHeadDistanceLayer

class simple_GINA(nn.Module):
    def __init__(self, input_length):
        super(simple_GINA, self).__init__()
        self.conv1 = nn.Conv1d(2,32, 3, padding=1)
        self.mp1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.mp2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.mp3 = nn.MaxPool1d(2)

        self.distance = MultiHeadDistanceLayer(2, 16, input_length//(2**3), 128)

        self.final = nn.Linear(input_length//(2**3)*2, 1)

    def forward(self, x):
        output = self.mp1(F.relu(self.conv1(x)))
        output = self.mp2(F.relu(self.conv2(output)))
        output = self.mp3(F.relu(self.conv3(output)))
        output = self.distance(output)
        output = output.view(output.size(0), -1)
        output = self.final(output)

        return output