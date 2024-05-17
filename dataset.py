"""datasets.py"""

import os
import numpy as np
import random
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

__datasets__ = ['cifar10', 'celeba']


class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"


def preprocess_dsprites(tensor):
    return (tensor.float().unsqueeze(0))*2-1


def preprocess_teapots(tensor):
    return (tensor.float().permute(2, 0, 1))/255*2-1


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2) 
        self.transform = transforms.Compose([ 
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),]) 
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2) 

        return img1, img2 

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class CustomImageFolder2(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder2, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = pil_loader(path1)
        img2 = pil_loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2
        #return img1, 1


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        #index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        #img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            #img2 = self.transform(img2)

        return img1, img1

    def __len__(self):
        return self.data_tensor.size(0)


class CustomTensorDataset3(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(60),
            transforms.Resize(64),
            transforms.ToTensor()
            ])

    def __getitem__(self, index1):
        #index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]*0.5+0.5
        #img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img1 = img1*2 -1
            #img2 = self.transform(img2)

        return img1, img1

    def __len__(self):
        return self.data_tensor.size(0)

class CustomTensorDataset2(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.num_values = 8
        self.color_values = torch.linspace(0, 1, self.num_values)

    def __getitem__(self, index1):
        img1 = self.data_tensor[index1]
        if self.transform is not None:
            img1 = self.transform(img1)*0.5+0.5

        r, g, b = img1.repeat(3, 1, 1).split(1, 0)
        r *= random.choice(self.color_values)
        g *= random.choice(self.color_values)
        b *= random.choice(self.color_values)
        img1 = torch.cat([r, g, b], 0)
        img1 = img1*2-1
        return img1, img1

    def __len__(self):
        return self.data_tensor.size(0)

#class CustomTensorDataset2(Dataset):
#    # red blue grenn dsprites
#    def __init__(self, data_tensor, transform=None):
#        self.data_tensor = data_tensor
#        self.transform = transform
#
#    def __getitem__(self, index1)
#        img1 = self.data_tensor[index1]
#        if self.transform is not None:
#            img1 = self.transform(img1)
#
#        color = random.choice([0,1,2])
#        r, g, b = img1.repeat(3, 1, 1).split(1, 0)
#        if color == 0: # Red
#            g = g*0-1
#            b = b*0-1
#        elif color == 1: # Green
#            r = r*0-1
#            b = b*0-1
#        else: # Blue
#            r = r*0-1
#            g = g*0-1
#
#        img1 = torch.cat([r, g, b], 0)
#        return img1, img1
#
#    def __len__(self):
#        return self.data_tensor.size(0)


#class CustomTensorDataset(Dataset):
#    def __init__(self, data_tensor, transform=None):
#        self.data_tensor = data_tensor
#        self.transform = transform
#
#    def __getitem__(self, index):
#        data = self.data_tensor[index]
#        if self.transform is not None:
#            data = self.transform(data)
#
#        return data, 1
#
#    def __len__(self):
#        return self.data_tensor.size(0)
#

def return_data(args):
    batch_size = args.batch_size
    num_workers = args.num_workers  

    transform = transforms.Compose([
        #transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    train_kwargs = {'root':'/home/rmapaij/HSpace-SAEs/datasets/celeba', 'transform':transform}
    dset = CustomImageFolder

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = dict()
    data_loader['train'] = train_loader

    return data_loader


if __name__ == '__main__':
    import argparse
    #os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_loader = return_data(args)
