from pathlib import Path
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from datasets.datasets import return_data
from utils.utils import cuda, where



class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion
        
    #Fast gradient sign method & One-step target class method
    def fgsm(self, x, y, targeted=False, epsilon=0.03, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)
        if targeted: #One-step target class method/ input y would have target y
            cost = self.criterion(h_adv, y)
        else: #Fast gradient sign method
            cost = -self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - epsilon*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)


        h = self.net(x)
        h_adv = self.net(x_adv)
        with torch.no_grad():
            adv_noise = x - x_adv

        return x_adv, h_adv, h, adv_noise
    
    #Basic iterative method
    def i_fgsm(self, x, y, targeted=False, epsilon=0.03, alpha=1, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        iters = int(min(255*eps + 4, 255*1.25*eps))
        for i in range(iters):
            h_adv = self.net(x_adv)
            if targeted:
                cost = self.criterion(h_adv, y)
            else:
                cost = -self.criterion(h_adv, y)

            self.net.zero_grad()

            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = where(x_adv > x+epsilon, x+epsilon, x_adv)
            x_adv = where(x_adv < x-epsilon, x-epsilon, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h

    #Iterative least-likely class method
    def IterativeLeastlikely(self,images, y, targeted=False, eps=0.03, alpha=1, x_val_min=-1, x_val_max=1):
        output = self.net(images)
        _, labels = torch.min(output.data, 1)
        labels = labels.detach_()
        clamp_max = 255
        # The paper said min(eps + 4, 1.25*eps) is used as iterations
        iters = int(min(255*eps + 4, 255*1.25*eps))
            
        scale = 0 
        if scale:
            eps = eps / 255
            clamp_max = clamp_max / 255
            
        for i in range(iters) :    
            images.requires_grad = True
            outputs = self.net(images)

            self.net.zero_grad()
            cost = self.criterion(outputs, labels)
            cost.backward()

            attack_images = images - alpha*images.grad.sign()
            
            a = torch.clamp(images - eps, min=0)
            b = (attack_images>=a).float()*attack_images + (a>attack_images).float()*a
            c = (b > images+eps).float()*(images+eps) + (images+eps >= b).float()*b
            images = torch.clamp(c, max=clamp_max).detach_()

        h_adv = self.net(images)
        return images, h_adv, output

        """
        h = self.net(x)
        _, net_pred = torch.min(h.data, dim=1)
        y_ll = net_pred
        y_ll = y_ll.detach_()
        x_adv = Variable(x.data, requires_grad=True)
        iter = int(min(255*epsilon+ 4, 255*1.25*epsilon))
        for i in range(iter):
            h_adv = self.net(x_adv)
            if targeted:
                cost = self.criterion(h_adv, y_ll)
            else:
                cost = self.criterion(h_adv, y_ll)
            self.net.zero_grad()

            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()
            
            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = where(x_adv > x+epsilon, x+epsilon, x_adv)
            x_adv = where(x_adv < x-epsilon, x-epsilon, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        h = self.net(x)
        h_adv = self.net(x_adv)
        with torch.no_grad():
            adv_noise = x_adv - x
        return x_adv, h_adv, h
        """
