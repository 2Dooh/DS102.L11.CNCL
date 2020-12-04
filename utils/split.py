import numpy as np
import torchvision.datasets as datasets
from sklearn.model_selection import StratifiedShuffleSplit 


def split(path):
    dataset = datasets.ImageFolder(root=path)
    X = np.arange(len(dataset)) # index from 0 -> dataset size
    y = np.array(dataset.targets) # label class
    print(dict(zip(*[res.tolist() for res in np.unique(y, return_counts=True)])))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return train_index, test_index

    



split('./data/CroppedImages32x32')