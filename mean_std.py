import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from tqdm.notebook import tqdm
from time import time
from PIL import Image


def pil_loader(path: str):
        """
        Converts image to RGB format.
        
        Parameters
        ----------
        path : str
            path to read image.
        Returns
        -------
        Image: PIL.Image.Image
            image converted to RGB
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

N_CHANNELS = 1

dataset = torchvision.datasets.ImageFolder(
                            root="./dataset/",
                            loader=pil_loader,
                            transform=transforms.ToTensor())
full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())

before = time()
mean = torch.zeros(1)
std = torch.zeros(1)
print('==> Computing mean and std..')
for inputs, _labels in tqdm(full_loader):
    for i in range(N_CHANNELS):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)

print("time elapsed: ", time()-before)


