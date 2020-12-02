from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class BigImageDataset(Dataset):
    def __init__(self, root, transform) -> None:
        self.root = root
        self.transform = transform
        self.img_names = sorted(os.listdir(self.root))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx) -> torch.TensorType:
        img_loc = os.path.join(self.root, self.img_names[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_img = self.transform(image)
        return tensor_img