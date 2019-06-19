from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from models.toynet import ToyNet_MNIST
from models.toynet import ToyNet_CIFAR10 # I added it.
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from datasets.datasets import return_data
from utils.utils import cuda, where
from adversary import Attack
from torchvision import transforms

# this class include triain/test/generate/ad_train/ad_test
class Solver(object):
    def __init__(self, args):
        self.args = args
        # Basic, load args value
        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.y_dim = args.y_dim # MNIST and CIFAR10 have class 10
        self.target = args.target # if you want to give pertubation to specific class then use it
        self.dataset = args.dataset
        self.data_loader = return_data(args)
        self.global_epoch = 0
        self.global_iter = 0
        self.print_ = not args.silent
        self.env_name = args.env_name # experiment name
        self.visdom = args.visdom # I have installed it but don't use it
        self.ckpt_dir = Path(args.ckpt_dir)
        self.save_ckpt_dir = Path('./checkpoints/' + args.env_name)
        print(self.save_ckpt_dir)
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if not self.save_ckpt_dir.exists():
            self.save_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)


        # Visualization Tools
        self.visualization_init(args)

        # Histories
        self.history = dict()
        self.history['acc'] = 0.
        self.history['epoch'] = 0
        self.history['iter'] = 0

        # Models & Optimizers
        self.model_init(args)
        self.load_ckpt = args.load_ckpt
        if args.load_ckpt_flag == True and self.load_ckpt != '':
            self.load_checkpoint(self.load_ckpt) #load trained weights

        # Adversarial Perturbation Generator
        #criterion = cuda(torch.nn.CrossEntropyLoss(), self.cuda)
        criterion = F.cross_entropy
        self.attack_mode = args.attack_mode
        if self.attack_mode == 'FGSM':
            self.attack = Attack(self.net, criterion=criterion)
        elif self.attack_mode == 'ILLC':
            self.attack = Attack(self.net, criterion=criterion)

    def visualization_init(self, args):
        # Visdom
        if self.visdom:
            from utils.visdom_utils import VisFunc
            self.port = args.visdom_port
            self.vf = VisFunc(enval=self.env_name, port=self.port)

    def model_init(self, args):
        # Network, defined in models folder, toynet.py and resnet.py
        if args.dataset =='MNIST':
            print("MNIST")
            self.net = cuda(ToyNet_MNIST(y_dim=self.y_dim), self.cuda)
        elif args.dataset =='CIFAR10':
            print("Dataset used CIFAR10")
            if args.network_choice == 'ToyNet':
                self.net = cuda(ToyNet_CIFAR10(y_dim=self.y_dim), self.cuda)
            elif args.network_choice == 'ResNet18':
                self.net = cuda(ResNet18(), self.cuda)
            elif args.network_choice == 'ResNet34':
                self.net = cuda(ResNet34(), self.cuda)
            elif args.network_choice == 'ResNet50':
                self.net = cuda(ResNet50(), self.cuda)
        self.net.weight_init(_type='kaiming')
        # setup optimizer
        self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}],
                                betas=(0.5, 0.999))
        
####################################################################################################
######                               Main functions defined                                    #####
####################################################################################################

    # train network with clean dataset
    def train(self):
        self.set_mode('train')
        acc_train_plt = [0]
        loss_plt = []
        acc_test_plt = [0]
        for e in range(self.epoch):
            self.global_epoch += 1
            local_iter = 0 
            correct = 0.
            cost = 0.
            total = 0.
            total_acc = 0.
            total_loss = 0.
            # train for each batch iteration
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1
                local_iter += 1
                #print("image size is ", np.shape(images))

                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))

                logit = self.net(x)
                prediction = logit.max(1)[1]

                correct = torch.eq(prediction, y).float().mean().data.item()
                cost = F.cross_entropy(logit, y)
                total_acc += correct
                total_loss += cost.data.item()

                self.optim.zero_grad()
                cost.backward() #back propagation
                self.optim.step()
                
                #for every 100th batch show accuracy and loss information of training result
                if batch_idx % 100 == 0:
                    if self.print_:
                        print()
                        print(self.env_name)
                        print('[{:03d}:{:03d}]'.format(self.global_epoch, batch_idx))
                        print('acc:{:.3f} loss:{:.3f}'.format(correct, cost.data.item()))
            total_acc = total_acc / local_iter
            total_loss = total_loss/local_iter
            acc_train_plt.append(total_acc)
            loss_plt.append(total_loss)
            acc_test_plt.append(self.test()) #show test results every epochs and save in acc_test_plt.
        print(" [*] Training Finished!")
        self.plot_result(acc_train_plt, acc_test_plt, loss_plt, self.history['acc']) 
        #plot results in graph and save to image, this function is defined in below

    # test network with clean images
    def test(self):
        self.set_mode('eval')
        correct = 0.
        cost = 0.
        total = 0.
        data_loader = self.data_loader['test']
        for batch_idx, (images, labels) in enumerate(data_loader):
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))

            logit = self.net(x)
            prediction = logit.max(1)[1]

            correct += torch.eq(prediction, y).float().sum().data.item()
            cost += F.cross_entropy(logit, y, size_average=False).data.item()
            total += x.size(0)
        accuracy = correct / total
        cost /= total

        if self.history['acc'] < accuracy:
            self.history['acc'] = accuracy
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            self.save_checkpoint('best_acc.tar')

        if self.print_:
            print()
            print('[{:03d}]\nTEST RESULT'.format(self.global_epoch))
            print('ACC:{:.4f}'.format(self.history['acc']))
            print('*TOP* ACC:{:.4f} at e:{:03d}'.format(self.history['acc'], self.global_epoch,))
            print()

        self.set_mode('train')
        return accuracy
    
    # generate perturbation images
    def generate(self, target, epsilon, alpha, iteration):
        self.set_mode('eval')
        x_true, y_true = self.sample_data() # take sample which size is batch_size
        if isinstance(target, int) and (target in range(self.y_dim)):
            y_target = torch.LongTensor(y_true.size()).fill_(target)
        else:
            y_target = None

        # generate perturbation images, inside of self.FGSM(defined below), there are fgsm and i-fgsm method
        # self.ILLC(defined below), there is illc methof
        if self.attack_mode == 'FGSM':
            x_adv, changed, values = self.FGSM(x_true, y_true, y_target, epsilon, alpha, iteration)
        elif self.attack_mode == 'ILLC':
            x_adv, changed, values = self.ILLC(x_true, y_true, y_target, epsilon, alpha, iteration)
        accuracy, cost, accuracy_adv, cost_adv = values

        # save the result image, you can find in outputs/experiment_name
        save_image(x_true,
                   self.output_dir.joinpath('legitimate(t:{},e:{},i:{}).jpg'.format(target,
                                                                                    epsilon,
                                                                                    iteration)),
                   nrow=10,
                   padding=2,
                   pad_value=0.5) # save clean images in one batch size
        save_image(x_adv,
                   self.output_dir.joinpath('perturbed(t:{},e:{},i:{}).jpg'.format(target,
                                                                                   epsilon,
                                                                                   iteration)),
                   nrow=10,
                   padding=2,
                   pad_value=0.5) # save perturbation(attacked) images in one batch size
        save_image(changed,
                   self.output_dir.joinpath('changed(t:{},e:{},i:{}).jpg'.format(target,
                                                                                 epsilon,
                                                                                 iteration)),
                   nrow=10,
                   padding=3,
                   pad_value=0.5) # save perturbation(attacked) images and classified result(by color) in one batch size

        if self.visdom:
            self.vf.imshow_multi(x_true.cpu(), title='legitimate', factor=1.5)
            self.vf.imshow_multi(x_adv.cpu(), title='perturbed(e:{},i:{})'.format(epsilon, iteration), factor=1.5)
            self.vf.imshow_multi(changed.cpu(), title='changed(white)'.format(epsilon), factor=1.5)

        print('[BEFORE] accuracy : {:.2f} cost : {:.3f}'.format(accuracy, cost))
        print('[AFTER] accuracy : {:.2f} cost : {:.3f}'.format(accuracy_adv, cost_adv))
        # show results of network performance to clean images and perturbated images

        self.set_mode('train')

    # adversarial training
    def ad_train(self, target, alpha, iteration, lamb):
        self.set_mode('train')
        acc_train_plt = [0]
        acc_test_plt = [0]
        loss_plt = []
        for e in range(self.epoch):
            self.global_epoch += 1
            local_iter = 0 
            correct = 0.
            cost = 0.
            total_acc = 0.
            total_loss = 0.
            total = 0.
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1
                local_iter += 1
                self.set_mode('eval')

                # compute k-number that how many of adversarial examples we will generate in each minibatch
                num_adv_image = self.batch_size//2
                
                # save images and labels which to generate to adversarial examples
                x_true = Variable(cuda(images[:num_adv_image], self.cuda))
                y_true = Variable(cuda(labels[:num_adv_image], self.cuda))
                
                # save clean images and labels
                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))

                if isinstance(target, int) and (target in range(self.y_dim)):
                    y_target = torch.LongTensor(y_true.size()).fill_(target)
                else:
                    y_target = None
                
                # get random value of epsilon through normal distribution in every batch iteration
                # if random epsilon value get overs to truncate value, set to 0.
                epsilon = abs(np.random.normal(0, 8 / 255))
                if epsilon > 16 / 255:
                    epsilon = 0
                
                # differ to attack method, generate adversarial examples
                if self.attack_mode == 'FGSM':
                    x[:num_adv_image], _, _ = self.FGSM(x_true, y_true, y_target, epsilon, alpha, iteration)
                elif self.attack_mode == 'ILLC':
                    x[:num_adv_image], _, _ = self.ILLC(x_true, y_true, y_target, epsilon, alpha, iteration)

                self.set_mode('train')
                logit = self.net(x)
                prediction = logit.max(1)[1]

                correct = torch.eq(prediction, y).float().mean().data.item()
                
                # The loss function is redefined with relative weight lambda as we have to more focus on clean data 
                # lambda is set to 0.3
                cost = (F.cross_entropy(logit[num_adv_image:], y[num_adv_image:]) \
                        + lamb*F.cross_entropy(logit[:num_adv_image], y[:num_adv_image]))*num_adv_image \
                        /(self.batch_size -(1-lamb)*num_adv_image)

                total_acc += correct
                total_loss += cost.data.item()
                self.optim.zero_grad()
                cost.backward()
                self.optim.step()

                if batch_idx % 100 == 0:
                    if self.print_:
                        print()
                        print(self.env_name)
                        print('[{:03d}:{:03d}]'.format(self.global_epoch, batch_idx))
                        print('acc:{:.3f} loss:{:.3f}'.format(correct, cost.data.item()))

            total_acc = total_acc / local_iter
            total_loss = total_loss / local_iter
            acc_train_plt.append(total_acc)
            loss_plt.append(total_loss)
            acc_test_plt.append(self.test())
            self.test()
        print(" [*] Training Finished!")
        self.plot_result(acc_train_plt, acc_test_plt, loss_plt, self.history['acc'])

    # generate all test dataset to adversarial images and test the performance of trained network
    def ad_test(self, target, epsilon, alpha, iteration):
        self.set_mode('eval')
        correct = 0.
        cost = 0.
        total = 0.
        data_loader = self.data_loader['test']
        for batch_idx, (images, labels) in enumerate(data_loader):
            x_true = Variable(cuda(images, self.cuda))
            y_true = Variable(cuda(labels, self.cuda))

            if isinstance(target, int) and (target in range(self.y_dim)):
                y_target = torch.LongTensor(y_true.size()).fill_(target)
            else:
                y_target = None

            if self.attack_mode == 'FGSM':
                x, _, _ = self.FGSM(x_true, y_true, y_target, epsilon, alpha, iteration)
            elif self.attack_mode == 'ILLC':
                x, _, _ = self.ILLC(x_true, y_true, y_target, epsilon, alpha, iteration)

            logit = self.net(x)
            prediction = logit.max(1)[1]

            correct += torch.eq(prediction, y_true).float().sum().data.item()
            cost += F.cross_entropy(logit, y_true, size_average=False).data.item()

            total += x.size(0)
        accuracy = correct / total
        cost /= total
        print('ACC:{:.4f}'.format(accuracy))
        self.set_mode('train')
     
###########################################end################################################
##############################################################################################
        
    #sample data which size is batch size
    def sample_data(self):
        data_loader = self.data_loader['test']
        for batch_idx, (images, labels) in enumerate(data_loader):
            x_true = Variable(cuda(images, self.cuda))
            y_true = Variable(cuda(labels, self.cuda))
            break
        return x_true, y_true

#############################################################################################
#####                           ILLC and FGSM defined                                   #####
#############################################################################################

    def ILLC(self, x, y_true, y_target=None, eps=0.03, alpha=2/255, iteration=1):
        self.set_mode('eval')
        x = Variable(cuda(x, self.cuda), requires_grad=True)
        y_true = Variable(cuda(y_true, self.cuda), requires_grad=False)

        if y_target is not None:
            targeted = True
            y_target = Variable(cuda(y_target, self.cuda), requires_grad=False)
        else:
            targeted = False

        # original image classification
        h = self.net(x)
        prediction = h.max(1)[1]
        accuracy = torch.eq(prediction, y_true).float().mean()

        cost = F.cross_entropy(h, y_true)

        # adversarial image classification
        if targeted:
            x_adv, h_adv, h = self.attack.IterativeLeastlikely(x, y_target, True, eps, alpha)
        else:
            x_adv, h_adv, h = self.attack.IterativeLeastlikely(x, y_true, False, eps, alpha)

        prediction_adv = h_adv.max(1)[1]
        accuracy_adv = torch.eq(prediction_adv, y_true).float().mean()
        cost_adv = F.cross_entropy(h_adv, y_true)

        # make indication of perturbed images that changed predictions of the classifier
        # it draw green and red boxes
        if targeted:
            changed = torch.eq(y_target, prediction_adv)
        else:
            changed = torch.eq(prediction, prediction_adv)
            changed = torch.eq(changed, 0)

        if self.dataset == 'MNIST':
            changed = changed.float().view(-1, 1, 1, 1).repeat(1, 3, 28, 28)
        elif self.dataset =='CIFAR10':
            changed = changed.float().view(-1, 1, 1, 1).repeat(1, 3, 32, 32)

        #fill the grid with color
        changed[:, 0, :, :] = where(changed[:, 0, :, :] == 1, 252, 91)
        changed[:, 1, :, :] = where(changed[:, 1, :, :] == 1, 39, 252)
        changed[:, 2, :, :] = where(changed[:, 2, :, :] == 1, 25, 25)
        changed = self.scale(changed/255)

        #fil the inner part of grid with image
        if self.dataset =='MNIST':
            changed[:, :, 3:-2, 3:-2] = x_adv.repeat(1, 3, 1, 1)[:, :, 3:-2, 3:-2]
        elif self.dataset =='CIFAR10':
            changed[:, :, 3:-2, 3:-2] = x_adv[:,:,3:-2,3:-2]

        self.set_mode('train')

        return x_adv.data, changed.data,\
                (accuracy.data.item(), cost.data.item(), accuracy_adv.data.item(), cost_adv.data.item())

    # Key point
    def FGSM(self, x, y_true, y_target=None, eps=0.03, alpha=2/255, iteration=1):
        self.set_mode('eval')
        x = Variable(cuda(x, self.cuda), requires_grad=True)
        y_true = Variable(cuda(y_true, self.cuda), requires_grad=False)

        if y_target is not None:
            targeted = True
            y_target = Variable(cuda(y_target, self.cuda), requires_grad=False)
        else:
            targeted = False

        # original image classification
        h = self.net(x)
        prediction = h.max(1)[1]
        accuracy = torch.eq(prediction, y_true).float().mean()

        cost = F.cross_entropy(h, y_true)

        # adversarial image classification
        if targeted:
            x_adv, h_adv, h = self.attack.i_fgsm(x, y_target, True, eps, alpha)
        else:
            x_adv, h_adv, h = self.attack.i_fgsm(x, y_true, False, eps, alpha)

        prediction_adv = h_adv.max(1)[1]
        accuracy_adv = torch.eq(prediction_adv, y_true).float().mean()
        cost_adv = F.cross_entropy(h_adv, y_true)

        # make indication of perturbed images that changed predictions of the classifier
        # it draw green and red boxes
        if targeted:
            changed = torch.eq(y_target, prediction_adv)
        else:
            changed = torch.eq(prediction, prediction_adv)
            changed = torch.eq(changed, 0)

        if self.dataset == 'MNIST':
            changed = changed.float().view(-1, 1, 1, 1).repeat(1, 3, 28, 28)
        elif self.dataset =='CIFAR10':
            changed = changed.float().view(-1, 1, 1, 1).repeat(1, 3, 32, 32)

        #fill the grid with color
        changed[:, 0, :, :] = where(changed[:, 0, :, :] == 1, 252, 91)
        changed[:, 1, :, :] = where(changed[:, 1, :, :] == 1, 39, 252)
        changed[:, 2, :, :] = where(changed[:, 2, :, :] == 1, 25, 25)
        changed = self.scale(changed/255)

        #fil the inner part of grid with image
        if self.dataset =='MNIST':
            changed[:, :, 3:-2, 3:-2] = x_adv.repeat(1, 3, 1, 1)[:, :, 3:-2, 3:-2]
        elif self.dataset =='CIFAR10':
            changed[:, :, 3:-2, 3:-2] = x_adv[:,:,3:-2,3:-2]

        self.set_mode('train')

        return x_adv.data, changed.data,\
                (accuracy.data.item(), cost.data.item(), accuracy_adv.data.item(), cost_adv.data.item())

 ##########################################################################################################
 ##########################################################################################################
    
    def save_checkpoint(self, filename='ckpt.tar'): #save weights
        model_states = {
            'net':self.net.state_dict(),
            }
        optim_states = {
            'optim':self.optim.state_dict(),
            }
        states = {
            'iter':self.global_iter,
            'epoch':self.global_epoch,
            'history':self.history,
            'args':self.args,
            'model_states':model_states,
            'optim_states':optim_states,
            }

        file_path = self.save_ckpt_dir / filename
        print(file_path)
        torch.save(states, file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'): #load weights
        file_path = self.ckpt_dir / filename
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))
            
    # change the model mode
    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else: raise('mode error. It should be either train or eval')

    # change 0~1 to -1~1 zero centered
    def scale(self, image):
        return image.mul(2).add(-1)
    
    def convert_torch2numpy(self, torch_img):
        np_img = np.transpose(torch_img.data.cpu().numpy(), (0,2,3,1))
        # PIL_image = transforms.ToPILImage()(transforms.ToTensor()(np_img),interpolation="bicubic")
        return np_img

    def plot_img(self, np_img, idx, title): #plot image
        plt.figure()
        plt.title(title)
        plt.imshow(np_img[idx], interpolation='nearest')

    def plot_result(self, acc_train_plt, acc_test_plt, loss_plt, best_acc, title='train_graph'): #save graph to image
        epoch = range(0, self.epoch+1)
        fig, ax1 = plt.subplots()
        ax1.plot(epoch, acc_train_plt, label='train_acc')
        ax1.plot(epoch, acc_test_plt, label='test_acc')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('accuracy')
        ax1.tick_params(axis='y')
        plt.legend(loc='upper left')
        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.plot(epoch[1:], loss_plt, linestyle="--", label='train_loss', color=color)
        ax2.set_ylabel('loss', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title("{}".format(self.env_name))
        plt.savefig('{}/{}/best_acc_{}.png'.format(self.args.output_dir, self.env_name, best_acc), dpi=350)
