# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os,argparse,time
import numpy as np
from omegaconf import OmegaConf

import torch

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import utils

tstart=time.time()


# Arguments
parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
# Load the config file
parser.add_argument('--config',  type=str, default='./configs/config_mnist5.yml')
flags =  parser.parse_args()
args = OmegaConf.load(flags.config)

print()


########################################################################################################################

# Args -- Experiment
if args.experiment=='pmnist':
    from dataloaders import pmnist as datagenerator
elif args.experiment=='mnist5':
    from dataloaders import mnist5 as datagenerator
elif args.experiment=='cifar100':
    from dataloaders import cifar100 as datagenerator
elif args.experiment=='miniimagenet':
    from dataloaders import miniimagenet as datagenerator
elif args.experiment=='multidatasets':
    from dataloaders import mulitidatasets as datagenerator
else:
    raise NotImplementedError

from acl import ACL as approach

# Args -- Network
if args.experiment == 'mnist5' or args.experiment == 'pmnist':
    from networks import mlp_acl as network
elif args.experiment == 'cifar100' or args.experiment == 'miniimagenet' or args.experiment == 'multidatasets':
    from networks import alexnet_acl as network
else:
    raise NotImplementedError

########################################################################################################################

def run(args, run_id):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

        # Faster run but not deterministic:
        # torch.backends.cudnn.benchmark = True

        # To get deterministic results that match with paper at cost of lower speed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data loader
    print('Instantiate data generators and model...')
    dataloader = datagenerator.DatasetGen(args)
    args.taskcla, args.inputsize = dataloader.taskcla, dataloader.inputsize
    if args.experiment == 'multidatasets': args.lrs = dataloader.lrs

    # Model
    net = network.Net(args)
    net = net.to(args.device)

    net.print_model_size()
    # print (net)

    # Approach
    appr=approach(net,args,network=network)

    # Loop tasks
    acc=np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)
    lss=np.zeros((len(args.taskcla),len(args.taskcla)),dtype=np.float32)

    for t,ncla in args.taskcla:

        print('*'*250)
        dataset = dataloader.get(t)
        print(' '*105, 'Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
        print('*'*250)

        # Train
        appr.train(t,dataset[t])
        print('-'*250)
        print()

        for u in range(t+1):
            # Load previous model and replace the shared module with the current one
            test_model = appr.load_model(u)
            test_res = appr.test(dataset[u]['test'], u, model=test_model)

            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, dataset[u]['name'],
                                                                                          test_res['loss_t'],
                                                                                          test_res['acc_t']))


            acc[t, u] = test_res['acc_t']
            lss[t, u] = test_res['loss_t']


        # Save
        print()
        print('Saved accuracies at '+os.path.join(args.checkpoint,args.output))
        np.savetxt(os.path.join(args.checkpoint,args.output),acc,'%.6f')

    # Extract embeddings to plot in tensorboard for miniimagenet
    if args.tsne == 'yes' and args.experiment == 'miniimagenet':
        appr.get_tsne_embeddings_first_ten_tasks(dataset, model=appr.load_model(t))
        appr.get_tsne_embeddings_last_three_tasks(dataset, model=appr.load_model(t))

    avg_acc, gem_bwt = utils.print_log_acc_bwt(args.taskcla, acc, lss, output_path=args.checkpoint, run_id=run_id)

    return avg_acc, gem_bwt



#######################################################################################################################


def main(args):
    utils.make_directories(args)
    utils.some_sanity_checks(args)
    utils.save_code(args)

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)


    accuracies, forgetting = [], []
    for n in range(args.num_runs):
        args.seed = n
        args.output = '{}_{}_tasks_seed_{}.txt'.format(args.experiment, args.ntasks, args.seed)
        print ("args.output: ", args.output)

        print (" >>>> Run #", n)
        acc, bwt = run(args, n)
        accuracies.append(acc)
        forgetting.append(bwt)


    print('*' * 100)
    print ("Average over {} runs: ".format(args.num_runs))
    print ('AVG ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean(), np.array(accuracies).std()))
    print ('AVG BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean(), np.array(forgetting).std()))


    print ("All Done! ")
    print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))
    utils.print_time()


#######################################################################################################################

if __name__ == '__main__':
    main(args)
