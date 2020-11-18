import torch
import numpy as np
from einops import repeat

a = np.load('./data/emat_calc/ecg_set.npy')
b = np.load('./data/emat_calc/pcg_set.npy')
c = np.load('./data/emat_calc/emat.npy', allow_pickle=True)
print(a.shape, b.shape, c.shape)
t = []
for index, i in enumerate(c):
    tmp = np.array(i)
    if tmp.shape[0] != 0:
        val = (tmp[:, 1] - tmp[:, 0]).mean()
        t.append(val)
    else:
        a = np.delete(a, index, axis=0)
        b = np.delete(b, index, axis=0)
        print(index)
    if val > 164:
        print(index)
    
print(a.shape, b.shape, c.shape)