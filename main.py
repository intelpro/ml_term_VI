import argparse
import numpy as np
import torch

from solver import Solver
from utils.utils import str2bool

def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    solver = Solver(args)
    
    # modes of main.py 
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    elif args.mode == 'generate':
        solver.generate(target=args.target,
                     epsilon=args.epsilon,
                     alpha=args.alpha,
                     iteration=args.iteration)
    elif args.mode == 'ad_train': #adversarial training, epsilon value randomly generated
        solver.ad_train(target=args.target,
                     alpha=args.alpha,
                     iteration=args.iteration,
                     lamb=0.3)
    elif args.mode == 'ad_test': #adversarial image test
        solver.ad_test(target=args.target,
                     epsilon=args.epsilon,
                     alpha=args.alpha,
                     iteration=args.iteration)
    else: return
    print('[*] Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_mode', required=True, type=str, choices=['FGSM', 'ILLC'])
    # FGSM for FGSM, One-step target, Basic iterative / ILLC for Iterative least-likely class method
    parser.add_argument('--epoch', type=int, default=30, help='epoch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--y_dim', type=int, default=10, help='the number of classes')
    parser.add_argument('--target', type=int, default=-1, help='target class for targeted generation')
    parser.add_argument('--dset_dir', type=str, default='datasets', help='dataset directory path')
    parser.add_argument('--summary_dir', type=str, default='summary', help='summary directory path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory path')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
    parser.add_argument('--load_ckpt', type=str, default='best_acc.tar', help='')
    parser.add_argument('--load_ckpt_flag', type=str2bool, required=True, help='load ckpt flag')
    parser.add_argument('--cuda', type=str2bool, default=True, help='enable cuda')
    parser.add_argument('--silent', type=str2bool, default=False, help='')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--iteration', type=int, default=1, help='the number of iteration for FGSM') 
    #For iterative methods, its value redefined later which depends on epsilon value. 
    parser.add_argument('--alpha', type=float, default=1, help='alpha for i-FGSM')
    parser.add_argument('--visdom', type=str2bool, default=False, help='enable visdom')
    parser.add_argument('--visdom_port', type=str, default=55558, help='visdom port')
    #######################################################################################################
    # These are the parameters we control.
    
    # Datasets(Cifar10, MNIST)
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset type')  # change to 'MNIST, 'CIFAR10'
    # Network we use(ToyNet for Mnist, ResNet18 for Cifar10)
    parser.add_argument('--network_choice', type=str, required=True, choices=['ToyNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101'])
    parser.add_argument('--env_name', type=str, default='ResNet_test', help='experiment name') # save folder name
    # These are the modes of main code. 'train' : train from vanila datasets, 'generate' : generate the perturbed images, 'test' : test from vanila datasets,
    #                                   'ad_train' : adversarial train with pertured images, don't need to set the epsilon
    #                                   'ad_test' : test for adversarial training
    
    parser.add_argument('--mode', type=str, default='train',
                                            choices=['train', 'generate', 'test', 'ad_train', 'ad_test'],
                                            help='train / test / generate / ad_train')
    # change to 100 when you make output figure because the output grid depends on this parameter
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')
    #Epsilon for adversarial noise, we give input value as 0 to 1 by divided 255(pixel/pixel) 
    parser.add_argument('--epsilon', type=float, default=128/255, help='epsilon for FGSM and i-FGSM') #change the magnitude of pertubation
    
    #######################################################################################################
    args = parser.parse_args()
    main(args)
