import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class simple_CNN(pl.LightningModule):
    def __init__(self, input_length):
        super(simple_CNN, self).__init__()
        self.conv1 = nn.Conv1d(2,32, 3, padding=1)
        self.mp1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.mp2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.mp3 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1)
        self.mp4 = nn.MaxPool1d(2)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, 3, padding=1)
        self.mp5 = nn.MaxPool1d(2)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 512, 3, padding=1)
        self.mp6 = nn.MaxPool1d(2)
        self.bn6 = nn.BatchNorm1d(512)
        
        self.final = nn.Linear(512*(input_length//(2**6)), 1)


    def forward(self, x):
        output = self.bn1(self.mp1(F.relu(self.conv1(x))))
        output = self.bn2(self.mp2(F.relu(self.conv2(output))))
        output = self.bn3(self.mp3(F.relu(self.conv3(output))))
        output = self.bn4(self.mp4(F.relu(self.conv4(output))))
        output = self.bn5(self.mp5(F.relu(self.conv5(output))))
        output = self.bn6(self.mp6(F.relu(self.conv6(output))))

        output = output.view(output.size(0), -1)
        output = self.final(output)

        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        predict = self(x)

        loss = F.l1_loss(predict.squeeze(), y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predict = self(x)
        loss = F.l1_loss(predict, y)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer