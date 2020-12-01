import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
from utils.cutout import Cutout
from utils.split import split


class CroppedImages:
    MEAN = [0.4698, 0.4907, 0.5077]
    STD = [0.2479, 0.2290, 0.2132]
    def __init__(self, 
                 folder,
                 num_workers, 
                 batch_size, 
                 pin_memory,
                 cutout=False,
                 cutout_length=None,
                 **kwargs):
        
        train_transform = transforms.Compose([
                                              transforms.RandomCrop((200, 150), padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.MEAN, self.STD)])
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))
        valid_transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize(self.MEAN, self.STD)])

        train_indices, test_indices = split(folder)

        dataset_train = datasets.ImageFolder(root=folder, transform=train_transform)
        dataset_test = datasets.ImageFolder(root=folder, transform=valid_transform)
        
        train_data = Subset(dataset_train, train_indices)

        test_data = Subset(dataset_test, test_indices)
        
        self.train_loader = DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       shuffle=True)
                                       
        # self.test_loader = DataLoader(dataset=test_data,
        #                               batch_size=batch_size,
        #                               pin_memory=pin_memory,
        #                               num_workers=num_workers,
        #                               shuffle=False)
        self.test_loader = None