import numpy as np
import csv
from PIL import Image
import sys
import os
import torch

from torchmeta.utils.data import Task, MetaDataset


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords



class CelebA(MetaDataset):
    """
    References
    ----------
    .. [1] Sitzmann, V., Martel, J. N., Bergman, A. W., Lindell, D. B., & Wetzstein, G.
        (2020). Implicit neural representations with periodic activation functions.
        arXiv preprint arXiv:2006.09661.
    """
    def __init__(self, num_samples_per_task, transform=None, target_transform=None,
                 dataset_transform=None):
        super(CelebA, self).__init__(meta_split='train',
            target_transform=target_transform, dataset_transform=dataset_transform)

        self.img_channels = 3
        self.root = '../data/img_align_celeba/img_align_celeba'
        self.fnames = []

        split = 'train'
        with open('../data/list_eval_partition.csv', newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rowreader:
                if split == 'train' and row[1] == '0':
                    self.fnames.append(row[0])
                elif split == 'val' and row[1] == '1':
                    self.fnames.append(row[0])
                elif split == 'test' and row[1] == '2':
                    self.fnames.append(row[0])

        self.num_samples_per_task = num_samples_per_task
        self.transform = transform


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # specify object for a single task

        selected_file = np.array(self.fnames)[index]

        # get num_tasks number of images
        task = CelebATask(index, self.num_samples_per_task, self.transform,
                          self.target_transform, self.root, self.fnames, selected_file)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class CelebATask(Task):
    # number of samples per task (number of celeba images to sample from)
    def __init__(self, index, num_samples, transform=None, target_transform=None,
                 root=None, fnames=None, selected_file=None):
        super(CelebATask, self).__init__(index, None)
        self.num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.fnames = fnames
        self.img_channels = 3
        self.rand_int = np.random.randint(0, 40)

        # self._idx = np.random.randint(0, len(self.fnames))
        # selected_file = np.array(self.fnames)[self._idx]

        # input should be (x,y) coordinate
        # output should be 3 channel pixel values

        # todo: set up sine test code plot(using model)
        # todo: set up inpainting tensorboard view while training
        # todo: use val for meta-val

        path = os.path.join(self.root, selected_file)
        img = Image.open(path)
        self._targets = self.transform(img) # [3, 32, 32]
        self._inputs_x = np.random.randint(0, 32, size=num_samples)
        self._inputs_y = np.random.randint(0, 32, size=num_samples)
        self.img = self._targets.permute(1, 2, 0).view(-1, self.img_channels)
        sidelength = (32,32)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input_x = self._inputs_x[index]
        input_y = self._inputs_y[index]
        target = self._targets[:, input_x, input_y]
        input = np.array([input_x, input_y])

        target = target.unsqueeze(0).float()
        target = torch.cat([self.img, target], axis=0)


        return (input, target)