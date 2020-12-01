import os
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.model_selection import StratifiedShuffleSplit 
from torch.utils.data.sampler import SubsetRandomSampler


def split(path):
    dataset = datasets.ImageFolder(root=path)
    X = np.arange(len(dataset)) # index from 0 -> dataset size
    y = np.array(dataset.targets) # label class
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return train_index, test_index

    



