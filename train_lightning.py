import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

from data.gen_data import gen_distance_peak_data, gen_distance_peak_data_choice
from model.simple_GINA import simple_GINA
from data.gen_data import DataGenerator

def test_single_data(model, x, y):
    model.eval()
    test_x = torch.tensor(x)
    test_y = torch.tensor(y)
    pred, attention = model(test_x)
    return pred, attention

if __name__ == '__main__':
    data_size = 1000
    epochs = 1000
    batch_size = 64
    """
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
    """

    x_train, y_train = gen_distance_peak_data_choice(100000)
    x_val, y_val = gen_distance_peak_data(100000)

    x_train = np.pad(x_train, ((0, 0), (0, 0), (450, 450)), mode='constant', constant_values=0)
    x_val = np.pad(x_val, ((0, 0), (0, 0), (450, 450)), mode='constant', constant_values=0)

    train_set = Data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_set = Data.TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    #test_set = Data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    val_loader = Data.DataLoader(val_set, batch_size, shuffle=True, num_workers=4)
    #test_loader = Data.DataLoader(test_set, batch_size, shuffle=True)
    model = simple_GINA(x_train.shape[2], 4).double()
    trainer = pl.Trainer(max_epochs=epochs, gpus='1', callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
    trainer.fit(model, train_loader, val_loader)
