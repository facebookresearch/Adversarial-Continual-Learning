# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys, time, os
import numpy as np
import torch
import copy
import utils

from copy import deepcopy
from tqdm import tqdm

sys.path.append('../')

from networks.discriminator import Discriminator

class ACL(object):

    def __init__(self, model, args, network):
        self.args=args
        self.nepochs=args.nepochs
        self.sbatch=args.batch_size

        # optimizer & adaptive lr
        self.e_lr=args.e_lr
        self.d_lr=args.d_lr

        if not args.experiment == 'multidatasets':
            self.e_lr=[args.e_lr] * args.ntasks
            self.d_lr=[args.d_lr] * args.ntasks
        else:
            self.e_lr = [self.args.lrs[i][1] for i in range(len(args.lrs))]
            self.d_lr = [self.args.lrs[i][1]/10. for i in range(len(args.lrs))]
            print ("d_lrs : ", self.d_lr)

        self.lr_min=args.lr_min
        self.lr_factor=args.lr_factor
        self.lr_patience=args.lr_patience

        self.samples=args.samples

        self.device=args.device
        self.checkpoint=args.checkpoint

        self.adv_loss_reg=args.adv
        self.diff_loss_reg=args.orth
        self.s_steps=args.s_step
        self.d_steps=args.d_step

        self.diff=args.diff

        self.network=network
        self.inputsize=args.inputsize
        self.taskcla=args.taskcla
        self.num_tasks=args.ntasks

        # Initialize generator and discriminator
        self.model=model
        self.discriminator=self.get_discriminator(0)
        self.discriminator.get_size()

        self.latent_dim=args.latent_dim

        self.task_loss=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_d=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_s=torch.nn.CrossEntropyLoss().to(self.device)
        self.diff_loss=DiffLoss().to(self.device)

        self.optimizer_S=self.get_S_optimizer(0)
        self.optimizer_D=self.get_D_optimizer(0)

        self.task_encoded={}

        self.mu=0.0
        self.sigma=1.0

        print()

    def get_discriminator(self, task_id):
        discriminator=Discriminator(self.args, task_id).to(self.args.device)
        return discriminator

    def get_S_optimizer(self, task_id, e_lr=None):
        if e_lr is None: e_lr=self.e_lr[task_id]
        optimizer_S=torch.optim.SGD(self.model.parameters(), momentum=self.args.mom,
                                    weight_decay=self.args.e_wd, lr=e_lr)
        return optimizer_S

    def get_D_optimizer(self, task_id, d_lr=None):
        if d_lr is None: d_lr=self.d_lr[task_id]
        optimizer_D=torch.optim.SGD(self.discriminator.parameters(), weight_decay=self.args.d_wd, lr=d_lr)
        return optimizer_D

    def train(self, task_id, dataset):
        self.discriminator=self.get_discriminator(task_id)

        best_loss=np.inf
        best_model=utils.get_model(self.model)


        best_loss_d=np.inf
        best_model_d=utils.get_model(self.discriminator)

        dis_lr_update=True
        d_lr=self.d_lr[task_id]
        patience_d=self.lr_patience
        self.optimizer_D=self.get_D_optimizer(task_id, d_lr)

        e_lr=self.e_lr[task_id]
        patience=self.lr_patience
        self.optimizer_S=self.get_S_optimizer(task_id, e_lr)


        for e in range(self.nepochs):

            # Train
            clock0=time.time()
            self.train_epoch(dataset['train'], task_id)
            clock1=time.time()

            train_res=self.eval_(dataset['train'], task_id)

            utils.report_tr(train_res, e, self.sbatch, clock0, clock1)

            # lowering the learning rate in the beginning if it predicts random chance for the first 5 epochs
            if (self.args.experiment == 'cifar100' or self.args.experiment == 'miniimagenet') and e == 4:
                random_chance=20.
                threshold=random_chance + 2

                if train_res['acc_t'] < threshold:
                    # Restore best validation model
                    d_lr=self.d_lr[task_id] / 10.
                    self.optimizer_D=self.get_D_optimizer(task_id, d_lr)
                    print("Performance on task {} is {} so Dis's lr is decreased to {}".format(task_id, train_res[
                        'acc_t'], d_lr), end=" ")

                    e_lr=self.e_lr[task_id] / 10.
                    self.optimizer_S=self.get_S_optimizer(task_id, e_lr)

                    self.discriminator=self.get_discriminator(task_id)

                    if task_id > 0:
                        self.model=self.load_checkpoint(task_id - 1)
                    else:
                        self.model=self.network.Net(self.args).to(self.args.device)


            # Valid
            valid_res=self.eval_(dataset['valid'], task_id)
            utils.report_val(valid_res)


            # Adapt lr for S and D
            if valid_res['loss_tot'] < best_loss:
                best_loss=valid_res['loss_tot']
                best_model=utils.get_model(self.model)
                patience=self.lr_patience
                print(' *', end='')
            else:
                patience-=1
                if patience <= 0:
                    e_lr/=self.lr_factor
                    print(' lr={:.1e}'.format(e_lr), end='')
                    if e_lr < self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer_S=self.get_S_optimizer(task_id, e_lr)

            if train_res['loss_a'] < best_loss_d:
                best_loss_d=train_res['loss_a']
                best_model_d=utils.get_model(self.discriminator)
                patience_d=self.lr_patience
            else:
                patience_d-=1
                if patience_d <= 0 and dis_lr_update:
                    d_lr/=self.lr_factor
                    print(' Dis lr={:.1e}'.format(d_lr))
                    if d_lr < self.lr_min:
                        dis_lr_update=False
                        print("Dis lr reached minimum value")
                        print()
                    patience_d=self.lr_patience
                    self.optimizer_D=self.get_D_optimizer(task_id, d_lr)
            print()

        # Restore best validation model (early-stopping)
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.discriminator.load_state_dict(copy.deepcopy(best_model_d))

        self.save_all_models(task_id)


    def train_epoch(self, train_loader, task_id):

        self.model.train()
        self.discriminator.train()

        for data, target, tt, td in train_loader:

            x=data.to(device=self.device)
            y=target.to(device=self.device, dtype=torch.long)
            tt=tt.to(device=self.device)

            # Detaching samples in the batch which do not belong to the current task before feeding them to P
            t_current=task_id * torch.ones_like(tt)
            body_mask=torch.eq(t_current, tt).cpu().numpy()
            # x_task_module=data.to(device=self.device)
            x_task_module=data.clone()
            for index in range(x.size(0)):
                if body_mask[index] == 0:
                    x_task_module[index]=x_task_module[index].detach()
            x_task_module=x_task_module.to(device=self.device)

            # Discriminator's real and fake task labels
            t_real_D=td.to(self.device)
            t_fake_D=torch.zeros_like(t_real_D).to(self.device)

            # ================================================================== #
            #                        Train Shared Module                          #
            # ================================================================== #
            # training S for s_steps
            for s_step in range(self.s_steps):
                self.optimizer_S.zero_grad()
                self.model.zero_grad()

                output=self.model(x, x_task_module, tt, task_id)
                task_loss=self.task_loss(output, y)

                shared_encoded, task_encoded=self.model.get_encoded_ftrs(x, x_task_module, task_id)
                dis_out_gen_training=self.discriminator.forward(shared_encoded, t_real_D, task_id)
                adv_loss=self.adversarial_loss_s(dis_out_gen_training, t_real_D)

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_encoded, task_encoded)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0

                total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss
                total_loss.backward(retain_graph=True)

                self.optimizer_S.step()

            # ================================================================== #
            #                          Train Discriminator                       #
            # ================================================================== #
            # training discriminator for d_steps
            for d_step in range(self.d_steps):
                self.optimizer_D.zero_grad()
                self.discriminator.zero_grad()

                # training discriminator on real data
                output=self.model(x, x_task_module, tt, task_id)
                shared_encoded, task_out=self.model.get_encoded_ftrs(x, x_task_module, task_id)
                dis_real_out=self.discriminator.forward(shared_encoded.detach(), t_real_D, task_id)
                dis_real_loss=self.adversarial_loss_d(dis_real_out, t_real_D)
                if self.args.experiment == 'miniimagenet':
                    dis_real_loss*=self.adv_loss_reg
                dis_real_loss.backward(retain_graph=True)

                # training discriminator on fake data
                z_fake=torch.as_tensor(np.random.normal(self.mu, self.sigma, (x.size(0), self.latent_dim)),dtype=torch.float32, device=self.device)
                dis_fake_out=self.discriminator.forward(z_fake, t_real_D, task_id)
                dis_fake_loss=self.adversarial_loss_d(dis_fake_out, t_fake_D)
                if self.args.experiment == 'miniimagenet':
                    dis_fake_loss*=self.adv_loss_reg
                dis_fake_loss.backward(retain_graph=True)

                self.optimizer_D.step()

        return


    def eval_(self, data_loader, task_id):
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t = 0, 0
        num=0
        batch=0

        self.model.eval()
        self.discriminator.eval()

        res={}
        with torch.no_grad():
            for batch, (data, target, tt, td) in enumerate(data_loader):
                x=data.to(device=self.device)
                y=target.to(device=self.device, dtype=torch.long)
                tt=tt.to(device=self.device)
                t_real_D=td.to(self.device)

                # Forward
                output=self.model(x, x, tt, task_id)
                shared_out, task_out=self.model.get_encoded_ftrs(x, x, task_id)
                _, pred=output.max(1)
                correct_t+=pred.eq(y.view_as(pred)).sum().item()

                # Discriminator's performance:
                output_d=self.discriminator.forward(shared_out, t_real_D, task_id)
                _, pred_d=output_d.max(1)
                correct_d+=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                # Loss values
                task_loss=self.task_loss(output, y)
                adv_loss=self.adversarial_loss_d(output_d, t_real_D)

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_out, task_out)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0

                total_loss = task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss

                loss_t+=task_loss
                loss_a+=adv_loss
                loss_d+=diff_loss
                loss_total+=total_loss

                num+=x.size(0)

        res['loss_t'], res['acc_t']=loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (batch + 1)
        res['loss_tot']=loss_total.item() / (batch + 1)
        res['size']=self.loader_size(data_loader)

        return res

    #

    def test(self, data_loader, task_id, model):
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t=0, 0
        num=0
        batch=0

        model.eval()
        self.discriminator.eval()

        res={}
        with torch.no_grad():
            for batch, (data, target, tt, td) in enumerate(data_loader):
                x=data.to(device=self.device)
                y=target.to(device=self.device, dtype=torch.long)
                tt=tt.to(device=self.device)
                t_real_D=td.to(self.device)

                # Forward
                output=model.forward(x, x, tt, task_id)
                shared_out, task_out=model.get_encoded_ftrs(x, x, task_id)

                _, pred=output.max(1)
                correct_t+=pred.eq(y.view_as(pred)).sum().item()

                # Discriminator's performance:
                output_d=self.discriminator.forward(shared_out, tt, task_id)
                _, pred_d=output_d.max(1)
                correct_d+=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_out, task_out)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0

                # Loss values
                adv_loss=self.adversarial_loss_d(output_d, t_real_D)
                task_loss=self.task_loss(output, y)

                total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss

                loss_t+=task_loss
                loss_a+=adv_loss
                loss_d+=diff_loss
                loss_total+=total_loss

                num+=x.size(0)

        res['loss_t'], res['acc_t']=loss_t.item() / (batch + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (batch + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (batch + 1)
        res['loss_tot']=loss_total.item() / (batch + 1)
        res['size']=self.loader_size(data_loader)

        return res



    def save_all_models(self, task_id):
        print("Saving all models for task {} ...".format(task_id+1))
        dis=utils.get_model(self.discriminator)
        torch.save({'model_state_dict': dis,
                    }, os.path.join(self.checkpoint, 'discriminator_{}.pth.tar'.format(task_id)))

        model=utils.get_model(self.model)
        torch.save({'model_state_dict': model,
                    }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))



    def load_model(self, task_id):

        # Load a previous model
        net=self.network.Net(self.args)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])

        # # Change the previous shared module with the current one
        current_shared_module=deepcopy(self.model.shared.state_dict())
        net.shared.load_state_dict(current_shared_module)

        net=net.to(self.args.device)
        return net


    def load_checkpoint(self, task_id):
        print("Loading checkpoint for task {} ...".format(task_id))

        # Load a previous model
        net=self.network.Net(self.args)
        checkpoint=torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))
        net.load_state_dict(checkpoint['model_state_dict'])
        net=net.to(self.args.device)
        return net


    def loader_size(self, data_loader):
        return data_loader.dataset.__len__()



    def get_tsne_embeddings_first_ten_tasks(self, dataset, model):
        from tensorboardX import SummaryWriter

        model.eval()

        tag_ = '_diff_{}'.format(self.args.diff)
        all_images, all_shared, all_private = [], [], []

        # Test final model on first 10 tasks:
        writer = SummaryWriter()
        for t in range(10):
            for itr, (data, _, tt, td) in enumerate(dataset[t]['tsne']):
                x = data.to(device=self.device)
                tt = tt.to(device=self.device)
                output = model.forward(x, x, tt, t)
                shared_out, private_out = model.get_encoded_ftrs(x, x, t)
                all_shared.append(shared_out)
                all_private.append(private_out)
                all_images.append(x)

        print (torch.stack(all_shared).size())

        tag = ['Shared10_{}_{}'.format(tag_,i) for i in range(1,11)]
        writer.add_embedding(mat=torch.stack(all_shared,dim=1).data, label_img=torch.stack(all_images,dim=1).data, metadata=list(range(1,11)),
                             tag=tag)#, metadata_header=list(range(1,11)))

        tag = ['Private10_{}_{}'.format(tag_, i) for i in range(1, 11)]
        writer.add_embedding(mat=torch.stack(all_private,dim=1).data, label_img=torch.stack(all_images,dim=1).data, metadata=list(range(1,11)),
                         tag=tag)#,metadata_header=list(range(1,11)))
        writer.close()


    def get_tsne_embeddings_last_three_tasks(self, dataset, model):
        from tensorboardX import SummaryWriter

        # Test final model on last 3 tasks:
        model.eval()
        tag = '_diff_{}'.format(self.args.diff)

        for t in [17,18,19]:
            all_images, all_labels, all_shared, all_private = [], [], [], []
            writer = SummaryWriter()
            for itr, (data, target, tt, td) in enumerate(dataset[t]['tsne']):
                x = data.to(device=self.device)
                y = target.to(device=self.device, dtype=torch.long)
                tt = tt.to(device=self.device)
                output = model.forward(x, x, tt, t)
                shared_out, private_out = model.get_encoded_ftrs(x, x, t)
                # print (shared_out.size())

                all_shared.append(shared_out)
                all_private.append(private_out)
                all_images.append(x)
                all_labels.append(y)

            writer.add_embedding(mat=torch.stack(all_shared,dim=1).data, label_img=torch.stack(all_images,dim=1).data,
                                 metadata=list(range(1,6)), tag='Shared_{}_{}'.format(t, tag))
                                 # ,metadata_header=list(range(1,6)))
            writer.add_embedding(mat=torch.stack(all_private,dim=1).data, label_img=torch.stack(all_images,dim=1).data,
                                 metadata=list(range(1,6)), tag='Private_{}_{}'.format(t, tag))
                                 # ,metadata_header=list(range(1,6)))

        writer.close()



        #
class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
