from model.simple_CNN import simple_CNN
import numpy as np
import torch
import torch.utils.data as Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from utils.generate_emat import cal_emat
import os

from model.GINA_periodic import GINA_periodic
from model.simple_CNN import simple_CNN

wandb_logger = WandbLogger(project='gina', log_model=True)

if __name__ == "__main__":
    batch_size = 64
    epochs = 100
    X_normal = np.load('./data/emat/normal_X.npy')
    X_abnormal = np.load('./data/emat/abnormal_X.npy')
    

    if os.path.exists('./data/emat/normal_y.npy'):
        y_normal = np.load('./data/emat/normal_y.npy')
    else:
        y_normal = cal_emat(X_normal[:, 0, :], X_normal[:, 1, :], fs=500)
        np.save('./data/emat/normal_y.npy', y_normal)
    
    if os.path.exists('./data/emat/abnormal_y.npy'):
        y_abnormal = np.load('./data/emat/abnormal_y.npy')
    else:
        y_abnormal = cal_emat(X_abnormal[:, 0, :], X_abnormal[:, 1, :], fs=500)
        np.save('./data/emat/abnormal_y.npy', y_abnormal)

    X = np.concatenate([X_normal, X_abnormal], axis=0)
    y = np.concatenate([y_normal, y_abnormal], axis=0)

    X = X[~np.isnan(y), ...]
    y = y[~np.isnan(y)]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

    
    train_set = Data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_set = Data.TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid))
    test_set = Data.TensorDataset(torch.tensor(X_valid[:64, ...]), torch.tensor(y_valid[:64, ...]))

    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    val_loader = Data.DataLoader(val_set, batch_size, shuffle=False, num_workers=4)
    test_loader = Data.DataLoader(test_set, batch_size, shuffle=False, num_workers=4)

    #model = GINA_periodic(X_train.shape[2], 16).double().cuda()
    model = simple_CNN(X_train.shape[2]).double().cuda()
    #model = simple_CNN(X_train.shape[2]).double()
    trainer = pl.Trainer(max_epochs=epochs, gpus=[0], callbacks=[EarlyStopping(monitor='val_loss', patience=10)], logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)
    
    trainer.test(test_dataloaders=test_loader)