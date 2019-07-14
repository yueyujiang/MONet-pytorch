import torch
import torch
from torch.autograd import Variable, Function
import matplotlib.pyplot as plt


from matplotlib.lines import Line2D
import numpy as np

def normal_KL_div_loss(logvar, mu):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# def KL_div_loss(masks1, masks2):
#     masks1_a = torch.log(1 - torch.exp(masks1) + 1e-10)
#     masks1 = torch.cat((masks1, masks1_a), 1)
#     masks2_a = torch.log(1 - torch.exp(masks2) + 1e-10)
#     masks2 = torch.cat((masks2, masks2_a), 1)
#     return torch.sum(torch.exp(masks1) * (masks1 - masks2))

def KL_div_loss(masks1, masks2):
    return torch.sum(torch.exp(masks1) * (masks1 - masks2))

def BCEloss(mask, target):
    return torch.sum(target * torch.log(mask) + (1 - target) * torch.log(mask))

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    fig = plt.figure()
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            try:
                ave_grads.append(p.grad.abs().mean())
            except:
                print('p')
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    return fig

def save_checkpoint(model, time_step, epochs, optimizer, checkpoint_PATH='./checkpoint'):
    torch.save({'epoch': epochs + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               checkpoint_PATH + '/m-' + str(epochs) + '-' + str(time_step) + '.pth.tar')

def load_checkpoint(model, checkpoint_PATH, optimizer=None):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        if optimizer:
            optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer

def cross_entropy(logmask, logtarget):
    target = torch.exp(logtarget)
    loss = torch.sum(target * logmask)
    return loss