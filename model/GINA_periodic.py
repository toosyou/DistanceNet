import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
#from model.distance import MultiHeadDistanceLayer
from model.GINA_V1_2_3 import MultiHeadDistanceLayer
from utils.visualization import atten_heatmap
import wandb

class GINA_periodic(pl.LightningModule):
    def __init__(self, input_length, num_head):
        super(GINA_periodic, self).__init__()
        self.conv1 = nn.Conv1d(2,32, 3, padding=1)
        self.mp1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.mp2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.mp3 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, 3, padding=1)
        self.mp4 = nn.MaxPool1d(2)
        self.bn4 = nn.BatchNorm1d(128)

        self.num_head = num_head

        self.distance = MultiHeadDistanceLayer(self.num_head, 16, input_length//(2**4), 128)

        self.final = nn.Linear(input_length//(2**4)*self.num_head, 1)

    def forward(self, x):
        output = self.bn1(self.mp1(F.relu(self.conv1(x))))
        output = self.bn2(self.mp2(F.relu(self.conv2(output))))
        output = self.bn3(self.mp3(F.relu(self.conv3(output))))
        output = self.bn4(self.mp4(F.relu(self.conv4(output))))
        output, attention = self.distance(output)
        output = output.reshape(output.size(0), -1)
        output = self.final(output)

        return output, attention
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # default forward
        output, atten = self(x)

        loss = F.l1_loss(output.squeeze(), y)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # default forward
        output, atten = self(x)

        loss = F.l1_loss(output.squeeze(), y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # default forward
        output, atten = self(x)

        for index in range(atten.size(0)):
            image_array = []
            for i in range(atten.size(1)):
                plot = atten_heatmap(x[index, 0, ::16].cpu(), x[index, 1, ::16].cpu(), atten[index, i, :, :].cpu(), figsize=(20,20))
                image_array.append(wandb.Image(plot))
            self.logger.experiment.log({f'plot_{index}': image_array})


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
