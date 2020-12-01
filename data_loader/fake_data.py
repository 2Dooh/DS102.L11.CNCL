import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class FakeData:
    def __init__(self, 
                 train_size,
                 test_size, 
                 input_size, 
                 n_classes, 
                 batch_size, 
                 pin_memory, 
                 num_workers, 
                 **kwargs):

        train_transform = transforms.Compose([transforms.RandomCrop(64, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        valid_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.FakeData(size=train_size,
                                      image_size=input_size,
                                      num_classes=n_classes,
                                      transform=train_transform)
        
        test_set = datasets.FakeData(size=test_size,
                                     image_size=input_size,
                                     num_classes=n_classes,
                                     transform=valid_transform)

        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)

        