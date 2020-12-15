# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from copy import deepcopy
import pickle
import time
import uuid
from subprocess import call
########################################################################################################################

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])


def report_tr(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
        'Diff loss:{:.3f} |'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'],
            1000 * sbatch * (time.time() - clock1) / res['size'], res['loss_tot'],
            res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')

def report_val(res):
    # Validation performance
    print(' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
        res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')


########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def save_print_log(taskcla, acc, lss, output_path):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()
    print ('ACC: {:5.4f}%'.format((np.mean(acc[acc.shape[0]-1,:]))))
    print()

    print ('BWD Transfer = ')

    print ()
    print ("Diagonal R_ii")
    for i in range(acc.shape[0]):
        print('\t',end='')
        print('{:5.2f}% '.format(np.diag(acc)[i]), end=',')


    print()
    print ("Last row")
    for i in range(acc.shape[0]):
        print('\t', end=',')
        print('{:5.2f}% '.format(acc[-1][i]), end=',')

    print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on our UCB paper (https://openreview.net/pdf?id=HklUCCVKDB)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    # print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')


    logs = {}
    # save results
    logs['name'] = output_path
    logs['taskcla'] = taskcla
    logs['acc'] = acc
    logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    with open(os.path.join(output_path, 'logs.p'), 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", os.path.join(output_path, 'logs.p'))


def print_log_acc_bwt(taskcla, acc, lss, output_path, run_id):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    print()
    print()
    # BWT calculated based on GEM paper (https://arxiv.org/abs/1706.08840)
    gem_bwt = sum(acc[-1]-np.diag(acc))/ (len(acc[-1])-1)
    # BWT calculated based on UCB paper (https://arxiv.org/abs/1906.02425)
    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT: {:5.2f}%'.format(gem_bwt))
    # print ('BWT (UCB paper): {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')


    logs = {}
    # save results
    logs['name'] = output_path
    logs['taskcla'] = taskcla
    logs['acc'] = acc
    logs['loss'] = lss
    logs['gem_bwt'] = gem_bwt
    logs['ucb_bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    path = os.path.join(output_path, 'logs_run_id_{}.p'.format(run_id))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", path)
    return avg_acc, gem_bwt


def print_running_acc_bwt(acc, task_num):
    print()
    acc = acc[:task_num+1,:task_num+1]
    avg_acc = np.mean(acc[acc.shape[0] - 1, :])
    gem_bwt = sum(acc[-1] - np.diag(acc)) / (len(acc[-1]) - 1)
    print('ACC: {:5.4f}%  || BWT: {:5.2f}% '.format(avg_acc, gem_bwt))
    print()


def make_directories(args):
    uid = uuid.uuid4().hex
    if args.checkpoint is None:
        os.mkdir('checkpoints')
        args.checkpoint = os.path.join('./checkpoints/',uid)
        os.mkdir(args.checkpoint)
    else:
        if not os.path.exists(args.checkpoint):
            os.mkdir(args.checkpoint)
        args.checkpoint = os.path.join(args.checkpoint, uid)
        os.mkdir(args.checkpoint)




def some_sanity_checks(args):
    # Making sure the chosen experiment matches with the number of tasks performed in the paper:
    datasets_tasks = {}
    datasets_tasks['mnist5']=[5]
    datasets_tasks['pmnist']=[10,20,30,40]
    datasets_tasks['cifar100']=[20]
    datasets_tasks['miniimagenet']=[20]
    datasets_tasks['multidatasets']=[5]


    if not args.ntasks in datasets_tasks[args.experiment]:
        raise Exception("Chosen number of tasks ({}) does not match with {} experiment".format(args.ntasks,args.experiment))

    # Making sure if memory usage is happenning:
    if args.use_memory == 'yes' and not args.samples > 0:
        raise Exception("Flags required to use memory: --use_memory yes --samples n where n>0")

    if args.use_memory == 'no' and args.samples > 0:
        raise Exception("Flags required to use memory: --use_memory yes --samples n where n>0")



def save_code(args):
    cwd = os.getcwd()
    des = os.path.join(args.checkpoint, 'code') + '/'
    if not os.path.exists(des):
        os.mkdir(des)

    def get_folder(folder):
        return os.path.join(cwd,folder)

    folders = [get_folder(item) for item in ['dataloaders', 'networks', 'configs', 'main.py', 'acl.py', 'utils.py']]

    for folder in folders:
        call('cp -rf {} {}'.format(folder, des),shell=True)


def print_time():
    from datetime import datetime

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Job finished at =", dt_string)

