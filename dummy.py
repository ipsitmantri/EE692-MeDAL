import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

model = models.resnet18(pretrained=True)
img = torch.zeros((1,3,224,224))
out = model.train()(img)
print(out.shape)