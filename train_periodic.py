import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from sklearn.model_selection import train_test_split
from utils.visualization import atten_heatmap
import os
import wandb

from data.gen_data import gen_distance_peak_data, gen_distance_peak_data_choice
from model.GINA_periodic import GINA_periodic
from data.gen_data import DataGenerator

wandb_logger = WandbLogger(project='gina', log_model=True)


def predict_plot(model, data):
    """
    Args:
        model: nn.Module
        data: (Array) with shape [#data, channel, data_length]
    """
    
    # default forward
    output, atten = model(torch.tensor(data).cuda())

    for index in range(atten.size(0)):
        for i in range(atten.size(1)):
            plot = atten_heatmap(data[index, 0, ::16], data[index, 1, ::16], atten[index, i, :, :].detach().cpu(), figsize=(5,5))
            wandb.log({f'{index}_plot_head_{i}': plot})
            plot.clf()

def distance_regression():
    data = np.load('./data/periodic_100000.npz')
    X, peaks = data['signals'], data['peaks']

    #X = X.swapaxes(1, 2)
    y = peaks[:, 1, :] - peaks[:, 0, :]
    y = y.mean(axis=-1)
    return X, y


if __name__ == "__main__":
    batch_size = 64
    epochs=1
    X, y = distance_regression()

    X = X[:10, ...]
    y = y[:10, ...]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

    
    train_set = Data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_set = Data.TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid))

    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    val_loader = Data.DataLoader(val_set, batch_size, shuffle=False, num_workers=4)

    model = GINA_periodic(X_train.shape[2], 16).double().cuda()
    trainer = pl.Trainer(max_epochs=epochs, gpus='1', callbacks=[EarlyStopping(monitor='val_loss', patience=2)], logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
    
    trainer.test(test_dataloaders=val_loader)

    #predict_plot(model, X_valid[:10, ...])