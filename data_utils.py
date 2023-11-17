import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hyperparams import params

def read_csv():
    df = pd.read_csv('assets/data/ML_data_daily.csv')
    df['dates'] = pd.date_range(start='1/1/2021', end='31/12/2021', freq='D')
    new_df = df.groupby(pd.Grouper(key = 'dates', freq = 'M'))['Receipt_Count'].sum().reset_index()
    return new_df

def split_sequence(sequence, n_steps, pad=False):
    X, y = list(), list()
    if pad:
        for i in range(n_steps-1, 0, -1):
            init = [0 for _ in range(n_steps)]
            for j in range(i, n_steps):
                init[j] = sequence[j-i]
            X.append(init)
            y.append(sequence[j-i+1])

    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return X, y

class ReceiptData(Dataset):
  def __init__(self, seq_receipts, labels, normalize_scale=1.):
    super().__init__()
    self.seq_receipts = np.array(seq_receipts) / normalize_scale
    self.labels = np.array(labels) / normalize_scale

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.seq_receipts[idx], self.labels[idx]
  
def dataset_pipeline(seq_len, receipts, pad=params['pad'], normalize=params['normalize']):
  seq_receipts, labels = split_sequence(receipts, seq_len, pad)

  train_receipts, test_receipts = seq_receipts[:-3], seq_receipts[-3:]
  train_labels, test_labels = labels[:-3], labels[-3:]

  train_dataset = ReceiptData(train_receipts, train_labels, normalize)
  train_loader = DataLoader(train_dataset, batch_size=len(train_receipts))

  test_dataset = ReceiptData(test_receipts, test_labels, normalize)
  test_loader = DataLoader(test_dataset, batch_size=len(test_receipts))

  return train_loader, test_loader