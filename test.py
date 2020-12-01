from utils.split import split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

path = './data/cropped_imgs'
train, test = split(path)
print(train)