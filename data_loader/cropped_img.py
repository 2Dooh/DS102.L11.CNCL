import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from utils.cutout import Cutout

class IntelImageClassification:
    IIC_MEAN = [.5] * 3
    IIC_STD = [.5] * 3
    def __init__(self, 
                 train_folder,
                 test_folder, 
                 num_workers, 
                 batch_size, 
                 pin_memory,
                 cutout=False,
                 cutout_length=None,
                 **kwargs):
        
        train_transform = transforms.Compose([transforms.RandomCrop(150, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.IIC_MEAN, self.IIC_STD)])
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))
        valid_transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize(self.IIC_MEAN, self.IIC_STD)])

        train_data = datasets.ImageFolder(root=train_folder,
                                          transform=train_transform)

        test_data = datasets.ImageFolder(root=test_folder,
                                         transform=valid_transform)
        
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