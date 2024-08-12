import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class UCRDataset(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


def load_data(data_root, dataset):
    """
    data_root: 'path of UCR/UEA folder'
    dataset: 'dataset'
    """

    train = pd.read_csv(os.path.join(data_root, dataset, dataset + '_TRAIN'), header=None)
    train_x = train.iloc[:, 1:]
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(data_root, dataset, dataset + '_TEST'), header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    train_test_dataset = pd.concat([train_x, test_x]).to_numpy(dtype=np.float32)
    train_test_target = pd.concat([train_target, test_target]).to_numpy(dtype=np.float32)

    num_classes = len(np.unique(train_test_target))

    return train_test_dataset, train_test_target, num_classes


def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def normalize_per_series(data):
    std_ = data.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    return (data - data.mean(axis=1, keepdims=True)) / std_


def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    test_set[ind_test] = np.take(col_mean, ind_test[1])
    return train_set, val_set, test_set
