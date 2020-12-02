import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.custom_datasets.big_image_dataset import BigImageDataset


class BigImages:
    MEAN = [0.4026, 0.4298, 0.4213]
    STD = [0.3144, 0.3023, 0.3056]
    def __init__(self, 
                 folder,
                 num_workers, 
                 batch_size, 
                 pin_memory,
                 **kwargs):
        
        transform = transforms.Compose([transforms.Resize((1080, 1920)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)])


        dataset = BigImageDataset(root=folder, transform=transform)
        
        self.test_loader = DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       shuffle=False)
        self.train_loader = None
                                       