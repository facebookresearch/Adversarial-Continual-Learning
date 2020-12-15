# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class Private(torch.nn.Module):
    def __init__(self, args):
        super(Private, self).__init__()

        self.ncha,self.size,_=args.inputsize
        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim
        self.num_tasks = args.ntasks
        self.nhid = args.units
        self.device = args.device

        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):
            self.linear = torch.nn.Sequential()
            self.linear.add_module('linear', torch.nn.Linear(self.ncha*self.size*self.size, self.latent_dim))
            self.linear.add_module('relu', torch.nn.ReLU(inplace=True))
            self.task_out.append(self.linear)

    def forward(self, x_p, task_id):
        x_p = x_p.view(x_p.size(0), -1)
        return self.task_out[task_id].forward(x_p)



class Shared(torch.nn.Module):

    def __init__(self,args):
        super(Shared, self).__init__()

        ncha,self.size,_=args.inputsize
        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim
        self.nhid = args.units
        self.nlayers = args.nlayers

        self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0.2)
        self.fc1=torch.nn.Linear(ncha*self.size*self.size, self.nhid)

        if self.nlayers == 3:
            self.fc2 = torch.nn.Linear(self.nhid, self.nhid)
            self.fc3=torch.nn.Linear(self.nhid,self.latent_dim)
        else:
            self.fc2 = torch.nn.Linear(self.nhid,self.latent_dim)

    def forward(self, x_s):

        h = x_s.view(x_s.size(0), -1)
        h = self.drop(self.relu(self.fc1(h)))
        h = self.drop(self.relu(self.fc2(h)))
        if self.nlayers == 3:
            h = self.drop(self.relu(self.fc3(h)))

        return h


class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        ncha,size,_=args.inputsize
        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim
        self.num_tasks = args.ntasks
        self.device = args.device

        if args.experiment == 'mnist5':
            self.hidden1 = 28
            self.hidden2 = 14
        elif args.experiment == 'pmnist':
            self.hidden1 = 28
            self.hidden2 = 28

        self.samples = args.samples

        self.shared = Shared(args)
        self.private = Private(args)

        self.head = torch.nn.ModuleList()
        for i in range(self.num_tasks):
            self.head.append(
                torch.nn.Sequential(
                    torch.nn.Linear(2 * self.latent_dim, self.hidden1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.hidden2, self.taskcla[i][1])
                ))

    def forward(self,x_s, x_p, tt, task_id):

        h_s = x_s.view(x_s.size(0), -1)
        h_p = x_s.view(x_p.size(0), -1)

        x_s = self.shared(h_s)
        x_p = self.private(h_p, task_id)

        x = torch.cat([x_p, x_s], dim=1)

        return torch.stack([self.head[tt[i]].forward(x[i]) for i in range(x.size(0))])


    def get_encoded_ftrs(self, x_s, x_p, task_id):
        return self.shared(x_s), self.private(x_p, task_id)


    def print_model_size(self):
        count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
        count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print('Num parameters in S       = %s ' % (self.pretty_print(count_S)))
        print('Num parameters in P       = %s,  per task = %s ' % (self.pretty_print(count_P),self.pretty_print(count_P/self.num_tasks)))
        print('Num parameters in p       = %s,  per task = %s ' % (self.pretty_print(count_H),self.pretty_print(count_H/self.num_tasks)))
        print('Num parameters in P+p     = %s ' % self.pretty_print(count_P+count_H))
        print('-------------------------->   Total architecture size: %s parameters (%sB)' % (self.pretty_print(count_S + count_P + count_H),
                                                                    self.pretty_print(4*(count_S + count_P + count_H))))

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
