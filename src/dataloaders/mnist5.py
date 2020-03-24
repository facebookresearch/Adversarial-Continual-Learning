# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from PIL import Image
import torch
import numpy as np
import os.path
import sys

import torch.utils.data as data
from torchvision import datasets, transforms



class iMNIST(datasets.MNIST):

    def __init__(self, root, classes, memory_classes, memory, task_num, train, transform=None, target_transform=None, download=True):

        super(iMNIST, self).__init__(root, task_num, transform=transform,
                                      target_transform=target_transform, download=download)

        self.train = train  # training set or test set
        self.root = root
        self.target_transform=target_transform
        self.transform=transform
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.data=np.array(self.data).astype(np.float32)
        self.targets=list(np.array(self.targets))

        self.train = train  # training set or test set
        if not isinstance(classes, list):
            classes = [classes]

        self.class_mapping = {c: i for i, c in enumerate(classes)}
        self.class_indices = {}

        for cls in classes:
            self.class_indices[self.class_mapping[cls]] = []


        data = []
        targets = []
        tt = []  # task module labels
        td = []  # discriminator labels

        for i in range(len(self.data)):
            if self.targets[i] in classes:
                data.append(self.data[i])
                targets.append(self.class_mapping[self.targets[i]])
                tt.append(task_num)
                td.append(task_num+1)
                self.class_indices[self.class_mapping[self.targets[i]]].append(i)


        if self.train:
            if memory_classes:
                for task_id in range(task_num):
                    for i in range(len(memory[task_id]['x'])):
                        if memory[task_id]['y'][i] in range(len(memory_classes[task_id])):
                            data.append(memory[task_id]['x'][i])
                            targets.append(memory[task_id]['y'][i])
                            tt.append(memory[task_id]['tt'][i])
                            td.append(memory[task_id]['td'][i])


        self.data = data.copy()
        self.targets = targets.copy()
        self.tt = tt.copy()
        self.td = td.copy()



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, tt, td = self.data[index], int(self.targets[index]), self.tt[index], self.td[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L')
        except:
            pass

        try:
            if self.transform is not None: img = self.transform(img)
        except:
            pass
        try:
            if self.target_transform is not None: tt = self.target_transform(tt)
            if self.target_transform is not None: td = self.target_transform(td)
        except:
            pass

        return img, target, tt, td


    def __len__(self):
        return len(self.data)




class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.seed = args.seed
        self.batch_size=args.batch_size
        self.pc_valid=args.pc_valid
        self.root = args.data_dir
        self.latent_dim = args.latent_dim

        self.num_tasks = args.ntasks
        self.num_classes = 10

        self.num_samples = args.samples

        self.inputsize = [1,28,28]
        mean = (0.1307,)
        std = (0.3081,)

        self.transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        self.taskcla = [[t, int(self.num_classes/self.num_tasks)] for t in range(self.num_tasks)]

        self.indices = {}
        self.dataloaders = {}
        self.idx={}

        self.num_workers = args.workers
        self.pin_memory = True

        np.random.seed(self.seed)
        self.task_ids = [[0,1], [2,3], [4,5], [6,7], [8,9]]

        self.train_set = {}
        self.test_set = {}

        self.task_memory = {}
        for i in range(self.num_tasks):
            self.task_memory[i] = {}
            self.task_memory[i]['x'] = []
            self.task_memory[i]['y'] = []
            self.task_memory[i]['tt'] = []
            self.task_memory[i]['td'] = []


    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        if task_id == 0:
            memory_classes = None
            memory=None
        else:
            memory_classes = self.task_ids
            memory = self.task_memory

        self.train_set[task_id] = iMNIST(root=self.root, classes=self.task_ids[task_id], memory_classes=memory_classes,
                                         memory=memory, task_num=task_id, train=True,
                                         download=True, transform=self.transformation)
        self.test_set[task_id] = iMNIST(root=self.root, classes=self.task_ids[task_id], memory_classes=None,
                                        memory=None, task_num=task_id, train=False,
                                        download=True, transform=self.transformation)

        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id], [len(self.train_set[task_id]) - split, split])


        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory, drop_last=True,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=int(self.batch_size * self.pc_valid),shuffle=True,
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory, drop_last=True,shuffle=True)

        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = '5Split-MNIST-{}-{}'.format(task_id,self.task_ids[task_id])

        print ("Training set size:      {}  images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size:    {}  images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Test set size:          {}  images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))
        return self.dataloaders


    def update_memory(self, task_id):
        num_samples_per_class = self.num_samples // len(self.task_ids[task_id])
        mem_class_mapping = {i: i for i, c in enumerate(self.task_ids[task_id])}

        # Looping over each class in the current task
        for i in range(len(self.task_ids[task_id])):

            dataset = iMNIST(root=self.root, classes=self.task_ids[task_id][i], memory_classes=None, memory=None,
                             task_num=task_id, train=True, download=True, transform=self.transformation)

            data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1,
                                                      num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory)

            # Randomly choosing num_samples_per_class for this class
            randind = torch.randperm(len(data_loader.dataset))[:num_samples_per_class]

            # Adding the selected samples to memory
            for ind in randind:
                self.task_memory[task_id]['x'].append(data_loader.dataset[ind][0])
                self.task_memory[task_id]['y'].append(mem_class_mapping[i])
                self.task_memory[task_id]['tt'].append(data_loader.dataset[ind][2])
                self.task_memory[task_id]['td'].append(data_loader.dataset[ind][3])