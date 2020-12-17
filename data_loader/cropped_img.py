import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
from utils.cutout import Cutout
from utils.split import split
from utils.resize_img import SquarePadding
import numpy as np


class CroppedImages:
    # MEAN = [0.3663, 0.3837, 0.3534]
    # STD = [0.3046, 0.2929, 0.2979]
    MEAN = [.5] * 3
    STD = [.5] * 3
    def __init__(self, 
                 train_folder,
                 test_folder,
                 num_workers, 
                 batch_size, 
                 pin_memory,
                 input_size,
                 class_labels,
                 padd_mode='constant',
                 cutout=False,
                 cutout_length=None,
                 **kwargs):
        
        train_transform = transforms.Compose([transforms.Resize(input_size[1:]),
                                              transforms.RandomCrop(input_size[1:], padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.MEAN, self.STD)])
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))
        valid_transform = transforms.Compose([transforms.Resize(input_size[1:]),
                                              transforms.ToTensor(), 
                                              transforms.Normalize(self.MEAN, self.STD)])

        # train_indices, test_indices = split(folder)

        train_data = datasets.ImageFolder(root=train_folder, transform=train_transform)
        test_data = datasets.ImageFolder(root=test_folder, transform=valid_transform)
        
        # train_data = Subset(test_data, train_indices)
        # test_data = Subset(test_data, test_indices)
        self.class_labels = class_labels

        self.train_loader = DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       shuffle=True)
                                       
        self.test_loader = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      pin_memory=pin_memory,
                                      num_workers=num_workers,
                                      shuffle=False)