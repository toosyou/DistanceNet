import torch
import torch.nn as nn
import numpy as np
from model import simple_CNN, Resnet
from data.gen_data import gen_distance_peak_data, gen_distance_peak_data_choice
import torch.utils.data as Data

def train(batch_size=64):
    model = simple_CNN.simple_CNN(4, 1, 1, 100)
    #model = Resnet.resnet18()
    model = model.cuda()
    model.train()

    num_data = 100000

    #X, y = gen_distance_peak_data(num_data=num_data, signal_length=200)
    X, y = gen_distance_peak_data_choice(num_data=num_data, signal_length=100)
    X = X[:, :1, :] + X[:, 1:, :]
    ds = Data.TensorDataset(torch.tensor(X-0.5, dtype=torch.float32), torch.tensor(np.abs(y), dtype=torch.float32))

    train_len = int(len(ds)*0.6)
    val_len = int(len(ds)*0.2)
    test_len = int(len(ds)*0.2)
    train_set, val_set, test_set = Data.random_split(ds, [train_len, val_len, test_len])
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(params=model.parameters())

    criterion = torch.nn.L1Loss()

    for epoch in range(1000):
        model.train()

        total_loss = 0
        avg_loss = 0
        fake_loss = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()

            pred = model(inputs)
            pred = torch.exp(pred)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss += avg_loss_calc(pred, labels).item()
            #fake_loss += avg_loss_calc(torch.zeros(labels.size()), labels).item()
        avg_val_loss = evaluate(model, val_loader)
        avg_test_loss = evaluate(model, test_loader)

        print(f"epoch: {epoch}, total_loss: {total_loss}, avg_loss: {avg_loss/batch_idx}, val_loss: {avg_val_loss}, test_loss: {avg_test_loss}")

def avg_loss_calc(pred, target):
    """
    Args:
        pred: (tensor) with sized [#data, ]
        target: (tensor) with sized [#data, ]
    """
    num_data = pred.size(0)
    total_loss = 0
    for i in range(num_data):
        total_loss += abs(pred[i] - target[i])

    return total_loss/num_data

def evaluate(model, val_loader):
    model.eval()
    avg_loss = 0
    for batch_idx, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        pred = model(inputs)
        avg_loss += avg_loss_calc(pred, labels).item()
    
    return avg_loss/batch_idx


if __name__ == "__main__":
    train()