import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.custom_datasets.big_image_dataset import BigImageDataset


class BigImages:
    # MEAN = [0.4026, 0.4298, 0.4213]
    # STD = [0.3144, 0.3023, 0.3056]
    
    # te
    # MEAN = [0.4698, 0.4907, 0.5077]
    # STD = [0.2479, 0.2290, 0.2132]

    MEAN = [.5] * 3
    STD = [.5] * 3
    # MEAN = [0.2274, 0.2428, 0.2381]
    # STD = [0.3063, 0.3085, 0.3071]

    # dang tot
    # MEAN = [0.3663, 0.3837, 0.3534]
    # STD = [0.3046, 0.2929, 0.2979]
    def __init__(self, 
                 folder,
                 num_workers, 
                 batch_size, 
                 pin_memory,
                 input_size,
                 **kwargs):
        
        transform = transforms.Compose([transforms.Resize(input_size[1:]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)])


        dataset = BigImageDataset(root=folder, transform=transform)
        
        self.test_loader = DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       shuffle=False)
        self.train_loader = None
                                       