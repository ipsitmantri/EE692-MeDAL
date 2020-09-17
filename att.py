import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import PIL
from matplotlib import pyplot as plt
from torchvision import datasets,transforms,models

mobile_net = models.mobilenet_v2(pretrained=True)
# f_out = open("file.txt","w")
# print(mobile_net,file=f_out)
# f_out.close()

class Attention(nn.Module):
    def __init__(self,pre_trained_net):
        super(Attention, self).__init__()
        self.L = 1000
        self.D = 128
        self.K = 1
        self.pre_trained_net = pre_trained_net
        self.l1 = nn.Sequential(nn.Linear(self.L,self.D),nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(self.L,self.D),nn.Sigmoid())
        self.l3 = nn.Linear(self.D,self.K)
        self.classifier = nn.Sequential(nn.Linear(self.L*self.K,16),nn.Softmax(dim=1))
    def forward(self,x):
        H = self.pre_trained_net(x)
        A_V = self.l1(H)
        A_U = self.l2(H)
        A = self.l3(A_V * A_U)
        A = torch.transpose(A,1,0)
        A = F.softmax(A,dim=1)
        M = torch.mm(A,H)
        Y = self.classifier(M)
        return Y
model = Attention(mobile_net)
x = torch.rand((10,3,224,224))
out = model(x)
print(torch.sum(out))
