import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets,models
from PIL import Image


# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity,self).__init__()
#
#     def forward(self,x):
#         return x
#
#
# model = models.mobilenet_v2(pretrained=True)
# model1 = models.mobilenet_v2(pretrained=True)
# a = model1.classifier
# model1.classifier = Identity()
# p = models.mobilenet_v2(pretrained=True)
# print(model.features[18])
# model.attention = nn.Sequential(nn.AvgPool2d(kernel_size=3))
# file = open('file.txt','w')
# print(model,file=file)
# file.close()


# model.classifier = Identity()
# print(model)
# class FCN32s(nn.Module):
#     def __init__(self,pretrained_net,n_class):
#         super(FCN32s,self).__init__()
#         self.n_class = n_class
#         self.pretrained_net = pretrained_net
#         self.relu = nn.ReLU(inplace=True)
#         self.deconv1 = nn.ConvTranspose2d(1280,512,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
#         self.bn1 = nn.BatchNorm2d(512)
#         self.deconv2 = nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.deconv3 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.deconv4 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.deconv5 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
#         self.bn5 = nn.BatchNorm2d(32)
#         self.classifier = nn.Conv2d(32,n_class,kernel_size=1)
#
#     def forward(self,x):
#         output = self.pretrained_net(x)
#         x5 = output
#         score = self.bn1(self.relu(self.deconv1(x5)))
#         score = self.bn2(self.relu(self.deconv2(score)))
#         score = self.bn3(self.relu(self.deconv3(score)))
#         score = self.bn4(self.relu(self.deconv4(score)))
#         score = self.bn5(self.relu(self.deconv5(score)))
#         score = self.classifier(score)
#         return score

# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention,self).__init__()
#         self.l1 = nn.Sequential(nn.Linear(1280,1280),nn.Tanh())
#         self.l2 = nn.Sequential(nn.Linear(1280,1280),nn.Sigmoid())
#         self.l3 = nn.Linear(1280,224)
#         self.fcn = FCN32s(p.features,3)
#     def forward(self,x):
#         y = self.fcn(x)
#         return y
#
# a = model.classifier
# model.classifier = Identity()
# model.attention = Attention()
# model.last = a
# b = Attention()
# x = torch.zeros(10,3,224,224)
# p = models.mobilenet_v2(pretrained=True)
# y = p.features(x)
# # z = a(y)
# print(y.shape)
# z = FCN32s(p.features,4)
# s = b(x)
# print(s.shape)