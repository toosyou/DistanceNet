import torch
import matplotlib.pyplot as plt
from train import train

def simulate():
    train_loss = []
    val_loss = []
    test_loss = []
    for i in range(1000, 100000):
        temp_train_loss, temp_val_loss, temp_test_loss = train(batch_size=64, data_size=i, train_epochs=100)
        train_loss.append(temp_train_loss)
        val_loss.append(temp_val_loss)
        test_loss.append(temp_test_loss)
    
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.plot(test_loss)
    plt.show()