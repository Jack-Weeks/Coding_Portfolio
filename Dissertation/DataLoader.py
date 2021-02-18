from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils, datasets
import nibabel as nib
import torchvision.transforms as tf
from PIL import Image
import random
from scipy.ndimage import zoom
# def data_loader(file_path):
#     structure_nii = nib.load(file_path)
#     image_array = np.asarray(structure_nii.get_fdata())
#
#     return image_array
# def load_data(root = '/Volumes/My Passport for Mac/Project/NewNift/Patches/train/error/'):
#     data_transform = transforms.Compose([transforms.ToTensor,
#                                          transforms.RandomHorizontalFlip])
#     RT_Dataset = datasets.DatasetFolder(root= root, loader= data_loader,
#                                         transform=data_transform, extensions=('.nii'))
#     dataset_loader = torch.utils.data.DataLoader(RT_Dataset, batch_size= 5)
#
#     return dataset_loader

# train_loader = load_data()
#
# for batchidx, img_label in enumerate(train_loader):
#     print(batchidx,img_label)

# def segmentation_Dataset(Dataset):
#     def __init__(self, nii_file):
#         self.nii_patch = nii_file
#         return
#     def __len__(self):
#         return len(self.nii_patch)
#     def __getitem__(self,idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         image =
#
#         return


def npy_loader(path):
    sample = np.load(path)
    sample = torch.Tensor(sample)
        #Random Horizontal Flip
    if random.random() > 0.5:
        sample = torch.flip(sample,[0,2])

        #Random vertical flipping
    if random.random() > 0.5:
        sample = torch.flip(sample,[1,2])
    return sample


# dataset = datasets.DatasetFolder(
#     root='NewNift/Patches/train2',
#     loader=npy_loader,
#     extensions='.npy'
# )
