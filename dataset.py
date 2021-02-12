import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class Dataset:
    """
    This class defines the methods to get train and test dataset
    """
    
    def __init__(self, transform=None):
        # load dataset using ImageFolder 
        if transform is None:
            # transform input images, resize/reshape every image to same shape.
            transform = transforms.Compose(
                [transforms.Resize((64,64)),
                    transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   

        self.dataset = torchvision.datasets.ImageFolder(
                            root="./dataset/",
                            loader=self.pil_loader,
                            transform=transform)

        # order the class
        self.dataset.class_to_index = {'without_mask': 0, 'with_mask': 1, 'non_humans': 2}

    
    def pil_loader(self, path: str):
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
        
    def get_dataset(self, train_split=None, test_split=None):
        """
        Gives training and testing data by splitting.
        Default
        -------
            splits train-test data into 70-30 ratio by default.
        Parameters
        ----------
        train_split : double
            dataset to split into from training data aspect, default is None
        test_split : double
            dataset to split into from test data aspect, default is None
        Returns
        -------
        train_dataset: tensor
            training data after splitting into training and testing
        test_dataset: tensor
            test data after splitting into training and testing
        """
        if train_split:
            # split data on training aspect ratio
            train_size = int(train_split*len(self.dataset))
            test_size = len(self.dataset) - train_size     

        elif test_split:
            # split data on testing aspect ratio
            test_size = int(test_split*len(self.dataset))
            train_size = len(self.dataset) - test_size      

        else:
            # default split into 70-30 ratio
            train_size = int(0.7*len(self.dataset))
            test_size = len(self.dataset) - train_size     

        #split the dataset into train, test
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        return train_dataset, test_dataset