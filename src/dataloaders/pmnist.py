# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys, os
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from utils import *


class PermutedMNIST(datasets.MNIST):

    def __init__(self, root, task_num, train=True, permute_idx=None, transform=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        self.data = torch.stack([img.float().view(-1)[permute_idx] for img in self.data])
        self.tl = (task_num) * torch.ones(len(self.data),dtype=torch.long)
        self.td = (task_num+1) * torch.ones(len(self.data),dtype=torch.long)


    def __getitem__(self, index):

        img, target, tl, td = self.data[index], self.targets[index], self.tl[index], self.td[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            print ("We are transforming")
            target = self.target_transform(target)

        return img, target, tl, td

    def __len__(self):
        return self.data.size(0)



class DatasetGen(object):

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.seed = args.seed
        self.batch_size=args.batch_size
        self.pc_valid=args.pc_valid
        self.num_samples = args.samples
        self.num_tasks = args.ntasks
        self.root = args.data_dir
        self.use_memory = args.use_memory

        self.inputsize = [1, 28, 28]
        mean = (0.1307,)
        std = (0.3081,)
        self.transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])

        self.taskcla = [[t, 10] for t in range(self.num_tasks)]

        self.train_set, self.test_set = {}, {}
        self.indices = {}
        self.dataloaders = {}
        self.idx={}
        self.get_idx()

        self.pin_memory = True
        self.num_workers = args.workers

        self.task_memory = []



    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        if task_id == 0:
            self.train_set[task_id] = PermutedMNIST(root=self.root, task_num=task_id, train=True,
                                            permute_idx=self.idx[task_id], transform=self.transformation)

            if self.use_memory == 'yes' and self.num_samples > 0:
                indices=torch.randperm(len(self.train_set[task_id]))[:self.num_samples]
                rand_subset=torch.utils.data.Subset(self.train_set[task_id], indices)
                self.task_memory.append(rand_subset)

        else:
            if self.use_memory == 'yes' and self.num_samples > 0:
                current_dataset = PermutedMNIST(root=self.root, task_num=task_id, train=True,
                                                permute_idx=self.idx[task_id], transform=self.transformation)
                d = []
                d.append(current_dataset)
                for m in self.task_memory:
                    d.append(m)
                self.train_set[task_id] = torch.utils.data.ConcatDataset(d)

                indices=torch.randperm(len(current_dataset))[:self.num_samples]
                rand_subset=torch.utils.data.Subset(current_dataset, indices)
                self.task_memory.append(rand_subset)

            else:
                self.train_set[task_id] = PermutedMNIST(root=self.root, task_num=task_id, train=True,
                                                permute_idx=self.idx[task_id], transform=self.transformation)

        self.test_set[task_id] = PermutedMNIST(root=self.root, task_num=task_id, train=False,
                                               permute_idx=self.idx[task_id], transform=self.transformation)

        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id],
                                                                 [len(self.train_set[task_id]) - split, split])

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size,
                                                num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=self.batch_size,
                                                num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size,
                                                num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)

        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = 'pmnist-{}'.format(task_id+1)

        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        return self.dataloaders


    def get_idx(self):
        for i in range(len(self.taskcla)):
            idx = list(range(self.inputsize[1] * self.inputsize[2]))
            self.idx[i] = shuffle(idx, random_state=self.seed * 100 + i)


