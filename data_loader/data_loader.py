import os
import torch
import pandas as pd
from torchvision import transforms, utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class InMemDataLoader(object):

    __initialized = False

    def __init__(
        self,
        tensors,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False
    ):
        """
        A torch dataloader that fetches data from memory.
        """
        tensors = [torch.tensor(tensor) for tensor in tensors]
        dataset = torch.utils.data.TensorDataset(*tensors)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(InMemDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield self.dataset[batch_indices]

    def __len__(self):
        return len(self.batch_sampler)

    def to(self, device):
        self.dataset.tensors = tuple(t.to(device) for t in self.dataset.tensors)
        return self


class PetsDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None
    ):
        datapoints_names = list(sorted(os.listdir(f'{root_dir}/datapoints')))
        label_names = list(sorted(os.listdir(f'{root_dir}/labels')))

        self.dataset = list(zip(datapoints_names, label_names))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datapoint_name = self.dataset[idx][0]
        label_name = self.dataset[idx][1]

        datapoint = Image.open(f'{self.root_dir}/datapoints/{datapoint_name}')
        label = Image.open(f'{self.root_dir}/labels/{label_name}')

        if self.transform:
            datapoint = self.transform(datapoint)
            label = self.transform(label)

        return datapoint, label


if __name__ == '__main__':
    current_dir = os.getcwd()

    dataset = PetsDataset(
        root_dir=f'{current_dir}/dataset/valid',
        transform=transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
        ])
    )

    print(len(dataset))

    for x, y in dataset:
        print(x.shape, y.shape)
        break

    dataloader = DataLoader(
        dataset, batch_size=4,
        shuffle=True, num_workers=0
    )

    print('\n=============================\n')

    for x, y in dataloader:
        print(x.shape, y.shape)
        break
