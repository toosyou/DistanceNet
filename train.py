import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import os

from data.gen_data import gen_distance_peak_data
from model.simple_GINA import simple_GINA
from data.gen_data import DataGenerator

def validation(model, loader):
    model.eval()
    criterion = nn.L1Loss(reduction="sum")
    total_loss = 0
    for x, y in loader:
        x = x.cuda().double()
        y = y.cuda()
        pred = model(x)
        pred = pred.squeeze()
        loss = criterion(pred, y)
        total_loss += loss.item()
    return total_loss

if __name__ == '__main__':
    data_size = 10000
    epochs = 100
    batch_size = 128
    
    g = DataGenerator(num_data=data_size, channel=2, signal_length=1000, padding_length=1000)
    g.addPeakShape(["triangle", "square"])
    if os.path.exists("./data/train.pkl"):
        x_train, y_train = torch.load("./data/train.pkl")
    else:
        x_train, y_train = g.generate(noisy_peak_num=4)
        torch.save((x_train, y_train), "./data/train.pkl")
    if os.path.exists("./data/val.pkl"):
        x_val, y_val = torch.load("./data/val.pkl")
    else:
        x_val, y_val = g.generate(noisy_peak_num=4)
        torch.save((x_val, y_val), "./data/val.pkl")

    train_set = Data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_set = Data.TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    #test_set = Data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_set, batch_size, shuffle=True)
    #test_loader = Data.DataLoader(test_set, batch_size, shuffle=True)

    model = simple_GINA(x_train.shape[2])
    model = model.double()
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    criterion = nn.L1Loss(reduction="sum")

    for epoch in range(epochs):
        print(f"epoch {epoch}: ", end="")
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_x = batch_x.double()

            pred = model(batch_x)
            pred = pred.squeeze()
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_validation_loss = validation(model, train_loader)
        print(f"training loss: {(total_loss/data_size)} validation loss: {(total_validation_loss/data_size)}")
