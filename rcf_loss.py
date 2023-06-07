import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torchinfo
import random
from BSDS500 import BSD, display_images
from rcf_model import RCF

class RCFLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        batch_size = X.shape[0]
        Yp = torch.sum(Y, dim=(1,2,3))
        Yn = torch.sum((1-Y), dim=(1,2,3))
        Yt = Yp+Yn
        copy = Y.clone()
        beta = copy * (Yn/(Yt)).view(batch_size, 1, 1, 1)
        alpha = torch.abs(copy-1) * 1.1 * (Yp/(Yt)).view(batch_size, 1, 1, 1)
        weights = beta + alpha
        return (F.binary_cross_entropy(X, Y, weight=weights, reduction='sum'))

#pixel to pixel accuracy (not f-score)
def accuracy(x, y):
    x = (x >= .5)
    x = x.type(torch.int)
    y = y.type(torch.int)
    ones = torch.ones(size=(5,1,321,481), device='cuda', dtype=torch.int32)
    temp = torch.bitwise_xor(x,y)
    temp2 = torch.bitwise_xor(temp, ones)
    sum = torch.sum(temp2)
    return sum / (5*321*481)