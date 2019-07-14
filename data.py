from __future__ import absolute_import
from torch.utils.data import Dataset
import os
import imageio
from PIL import Image
import numpy as np
import torch
import random

class MultiDSprites(Dataset):
    def __init__(self, root='../../../datasets/Multi-dSprites',
                 train=False, val=False, test=False):
        root = os.path.expanduser(root)
        if train:
            filename = 'img_color.npz'
        else:
            raise AssertionError
        if filename.endswith('.pt'):
            self.data = torch.load(os.path.join(root, filename))
        elif filename.endswith('.npz'):
            data_np = np.load(os.path.join(root, filename))
            self.data = torch.from_numpy(data_np['arr_0']).permute(0, 3, 1, 2).float()

        n = self.data.size(0)
        self.train_data = self.data[: int(n*3/4)]
        self.test_data = self.data[int(n*3/4):]
        self.train = train

    def __getitem__(self, index):
        if self.train:
            d = self.train_data[index]
        else:
            d = self.test_data[index]
        bg = torch.ones(d.shape)
        mask = torch.ones((d.shape[1], d.shape[2]))
        for i in range(3):
            bg[i, :, :] *= random.random()
            mask[d[i, :, :]!=0] = 0
        d += bg * mask
        return d

    def __len__(self):
        if self.train:
            return self.train_data.size(0)
        else:
            return self.test_data.size(0)

class Dsprite(Dataset):
    def __init__(self, root='../../../datasets/Multi-dSprites'):
        filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        dataset_zip = np.load(os.path.join(root, filename), allow_pickle=True, encoding='latin1')
        self.imgs = dataset_zip['imgs']

    def __getitem__(self, index):
        x = torch.from_numpy(self.imgs[index]).float()
        x = x.view(1, x.shape[0], -1)
        return x

    def __len__(self):
        return len(self.imgs)
