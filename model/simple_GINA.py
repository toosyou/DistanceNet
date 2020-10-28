import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model.distance import MultiHeadDistanceLayer

class simple_GINA(pl.LightningModule):
    def __init__(self, input_length, num_head):
        super(simple_GINA, self).__init__()
        self.conv1 = nn.Conv1d(2,32, 3, padding=1)
        self.mp1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.mp2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.mp3 = nn.MaxPool1d(2)

        self.num_head = num_head

        self.distance = MultiHeadDistanceLayer(self.num_head, 16, input_length//(2**3), 128)

        self.final = nn.Linear(input_length//(2**3)*self.num_head, 1)

    def forward(self, x):
        output = self.mp1(F.relu(self.conv1(x)))
        output = self.mp2(F.relu(self.conv2(output)))
        output = self.mp3(F.relu(self.conv3(output)))
        output, attention = self.distance(output)
        output = output.reshape(output.size(0), -1)
        output = self.final(output)

        return output, attention
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # default forward
        output, atten = self.forward(x)

        loss = F.l1_loss(output.squeeze(), y)

        self.log('train_loss', loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # default forward
        output, atten = self.forward(x)

        loss = F.l1_loss(output.squeeze(), y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
