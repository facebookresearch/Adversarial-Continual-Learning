# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import utils

class Shared(torch.nn.Module):

    def __init__(self,args):
        super(Shared, self).__init__()

        self.ncha,size,_=args.inputsize
        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim

        if args.experiment == 'cifar100':
            hiddens = [64, 128, 256, 1024, 1024, 512]

        elif args.experiment == 'miniimagenet':
            hiddens = [64, 128, 256, 512, 512, 512]

            # ----------------------------------
        elif args.experiment == 'multidatasets':
            hiddens = [64, 128, 256, 1024, 1024, 512]

        else:
            raise NotImplementedError

        self.conv1=torch.nn.Conv2d(self.ncha,hiddens[0],kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(hiddens[0],hiddens[1],kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(hiddens[1],hiddens[2],kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(hiddens[2]*s*s,hiddens[3])
        self.fc2=torch.nn.Linear(hiddens[3],hiddens[4])
        self.fc3=torch.nn.Linear(hiddens[4],hiddens[5])
        self.fc4=torch.nn.Linear(hiddens[5], self.latent_dim)


    def forward(self, x_s):
        x_s = x_s.view_as(x_s)
        h = self.maxpool(self.drop1(self.relu(self.conv1(x_s))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h = h.view(x_s.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        h = self.drop2(self.relu(self.fc3(h)))
        h = self.drop2(self.relu(self.fc4(h)))
        return h



class Private(torch.nn.Module):
    def __init__(self, args):
        super(Private, self).__init__()

        self.ncha,self.size,_=args.inputsize
        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim
        self.num_tasks = args.ntasks
        self.device = args.device

        if args.experiment == 'cifar100':
            hiddens=[32,32]
            flatten=1152

        elif args.experiment == 'miniimagenet':
            # hiddens=[8,8]
            # flatten=1800
            hiddens=[16,16]
            flatten=3600


        elif args.experiment == 'multidatasets':
            hiddens=[32,32]
            flatten=1152


        else:
            raise NotImplementedError


        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):
            self.conv = torch.nn.Sequential()
            self.conv.add_module('conv1',torch.nn.Conv2d(self.ncha, hiddens[0], kernel_size=self.size // 8))
            self.conv.add_module('relu1', torch.nn.ReLU(inplace=True))
            self.conv.add_module('drop1', torch.nn.Dropout(0.2))
            self.conv.add_module('maxpool1', torch.nn.MaxPool2d(2))
            self.conv.add_module('conv2', torch.nn.Conv2d(hiddens[0], hiddens[1], kernel_size=self.size // 10))
            self.conv.add_module('relu2', torch.nn.ReLU(inplace=True))
            self.conv.add_module('dropout2', torch.nn.Dropout(0.5))
            self.conv.add_module('maxpool2', torch.nn.MaxPool2d(2))
            self.task_out.append(self.conv)
            self.linear = torch.nn.Sequential()

            self.linear.add_module('linear1', torch.nn.Linear(flatten,self.latent_dim))
            self.linear.add_module('relu3', torch.nn.ReLU(inplace=True))
            self.task_out.append(self.linear)


    def forward(self, x, task_id):
        x = x.view_as(x)
        out = self.task_out[2*task_id].forward(x)
        out = out.view(out.size(0),-1)
        out = self.task_out[2*task_id+1].forward(out)
        return out



class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.ncha,size,_=args.inputsize
        self.taskcla=args.taskcla
        self.latent_dim = args.latent_dim
        self.num_tasks = args.ntasks
        self.samples = args.samples
        self.image_size = self.ncha*size*size
        self.args=args

        self.hidden1 = args.head_units
        self.hidden2 = args.head_units//2

        self.shared = Shared(args)
        self.private = Private(args)

        self.head = torch.nn.ModuleList()
        for i in range(self.num_tasks):
            self.head.append(
                torch.nn.Sequential(
                    torch.nn.Linear(2*self.latent_dim, self.hidden1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(),
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(self.hidden2, self.taskcla[i][1])
                ))


    def forward(self, x_s, x_p, tt, task_id):

        x_s = x_s.view_as(x_s)
        x_p = x_p.view_as(x_p)

        x_s = self.shared(x_s)
        x_p = self.private(x_p, task_id)

        x = torch.cat([x_p, x_s], dim=1)

        if self.args.experiment == 'multidatasets':
            # if no memory is used this is faster:
            y=[]
            for i,_ in self.taskcla:
                y.append(self.head[i](x))
            return y[task_id]
        else:
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
        print('Num parameters in P+p    = %s ' % self.pretty_print(count_P+count_H))
        print('-------------------------->   Architecture size: %s parameters (%sB)' % (self.pretty_print(count_S + count_P + count_H),
                                                                    self.pretty_print(4*(count_S + count_P + count_H))))

        print("-------------------------->   Memory size: %s samples per task (%sB)" % (self.samples,
                                                                                        self.pretty_print(self.num_tasks*4*self.samples*self.image_size)))
        print("------------------------------------------------------------------------------")
        print("                               TOTAL:  %sB" % self.pretty_print(4*(count_S + count_P + count_H)+self.num_tasks*4*self.samples*self.image_size))

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

