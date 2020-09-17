import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models,transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def onehot(file,arg):
    filew = pd.read_csv(file)
    # one_hots = np.empty((filew.shape[0], 11),dtype=object)
    one_hots = {}
    for i in range(filew.shape[0]):
        if arg == 'patho_1':
            primary_score = filew.loc[i][0][-7]
            secondary_score = filew.loc[i][0][-5]
        elif arg == 'patho_2':
            primary_score = filew.loc[i][0][-3]
            secondary_score = filew.loc[i][0][-1]
        # one_hots[i,0] = filew.loc[i][0][:-4]
        key = filew.loc[i][0][:-8]
        if primary_score == '0' and secondary_score == '0':
            # one_hots[i, 1:] = np.eye(10)[0, :]
            one_hots[key] = np.eye(10)[0,:]
        elif primary_score == '1' and secondary_score == '1':
            # one_hots[i, 1:] = np.eye(10)[1, :]
            one_hots[key] = np.eye(10)[1, :]
        elif primary_score == '2' and secondary_score == '2':
            # one_hots[i, 1:] = np.eye(10)[2, :]
            one_hots[key] = np.eye(10)[2, :]
        elif primary_score == '3' and secondary_score == '3':
            # one_hots[i, 1:] = np.eye(10)[3, :]
            one_hots[key] = np.eye(10)[3, :]
        elif primary_score == '1' and secondary_score == '2':
            # one_hots[i, 1:] = np.eye(10)[4, :]
            one_hots[key] = np.eye(10)[4, :]
        elif primary_score == '1' and secondary_score == '3':
            # one_hots[i, 1:] = np.eye(10)[5, :]
            one_hots[key] = np.eye(10)[5, :]
        elif primary_score == '2' and secondary_score == '1':
            # one_hots[i, 1:] = np.eye(10)[6, :]
            one_hots[key] = np.eye(10)[6, :]
        elif primary_score == '2' and secondary_score == '3':
            # one_hots[i, 1:] = np.eye(10)[7, :]
            one_hots[key] = np.eye(10)[7, :]
        elif primary_score == '3' and secondary_score == '1':
            # one_hots[i, 1:] = np.eye(10)[8, :]
            one_hots[key] = np.eye(10)[8, :]
        elif primary_score == '3' and secondary_score == '2':
            # one_hots[i, 1:] = np.eye(10)[9, :]
            one_hots[key] = np.eye(10)[9, :]
        else:
            print('you are wrong, there are more than 10 possibilities,primary score:{},secondary score:{}'.format(primary_score,secondary_score))
            # one_hots[i, :] = np.zeros(10)[0, :]
            one_hots[key] = np.eye(10)[0, :]
    return one_hots
def preprocess(image):
    transform = transforms.Compose([
        # transforms.RandomRotation(30),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(),
        # transforms.RandomCrop([224, 224]),
        transforms.Resize(size=224,interpolation=2)
    ])
    return transform(image)

class Attention(nn.Module):

    def __init__(self, pre_trained_net):
        super(Attention, self).__init__()
        self.L = 1000
        self.D = 128
        self.K = 1
        self.pre_trained_net = pre_trained_net
        self.l1 = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.l2 = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.l3 = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, self.L * self.K), nn.ReLU())
        self.final = nn.Sequential(nn.Linear(self.L * self.K, 10),nn.Softmax(dim=1))
        self.classifier_debug = nn.Sequential(nn.Linear(self.L, 10), nn.Softmax(dim=1))

    def forward(self, x):
        H = self.pre_trained_net(x)
        A_V = self.l1(H)
        A_U = self.l2(H)
        A = self.l3(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        Y = self.classifier(M)
        Y = self.final(Y)
        # Y = self.classifier_debug(H)
        return Y


preTrainedModel = models.resnet18(pretrained=True)
model = Attention(preTrainedModel)
model.load_state_dict(torch.load('best_wts.pth'))
# print(model.parameters)

PARENT_DIR = 'dataset_TMA'
test_dir = os.path.join(PARENT_DIR,'test_patches_750')
patho_1 = os.path.join(test_dir,'patho_1')
patho_2 = os.path.join(test_dir,'patho_2')
# print(len(os.listdir(patho_1)))
# print(len(os.listdir(patho_2)))
tma_info = os.path.join(PARENT_DIR, 'tma_info')
test_arrays = ['ZT80']

one_hots_1 = onehot('dataset_TMA/tma_info/ZT80_gleason_scores.csv','patho_1')
one_hots_2 = onehot('dataset_TMA/tma_info/ZT80_gleason_scores.csv','patho_2')
# print(one_hots_1)
# print(one_hots_2)
model.cuda()
model.eval()
invalids = {}
valids = []
all_outputs = []
torch.set_grad_enabled(False)
for dir in tqdm(range(len(os.listdir(patho_1)))):
    paths = os.path.join(patho_1,os.listdir(patho_1)[dir])
    if len(os.listdir(paths)) == 0:
        # invalids.append(paths.split('\\')[3])
        invalids[dir] = paths.split('\\')[3]

    else:
        bag = np.zeros((len(os.listdir(paths)), 224, 224, 3), dtype=np.uint8)
        Y = one_hots_1[os.listdir(patho_1)[dir]]
        for i in range(len(os.listdir(paths))):
            path = os.listdir(paths)[i]
            img = Image.open(os.path.join(paths, path))
            img = preprocess(img)
            bag[i, :, :, :] = img
        bag = torch.from_numpy(bag).type(torch.float32)
        bag = bag.cuda()
        Y = torch.from_numpy(Y).type(torch.LongTensor).view(1, 10).cuda()
        _, target = torch.max(Y, 1)
        target = target.cuda()
        outputs_val = model(bag.view(-1, 3, 224, 224))
        all_outputs.append(outputs_val)
        valids.append(one_hots_1[paths.split('\\')[3]])
print(invalids)
# torch.save(all_outputs,'all_outputs_3.pt')
# outputs = torch.load('all_outputs_3.pt')

# for i in range(len(outputs)):
#     if outputs[i] == 1 + torch.zeros((10)):
#         print(one_hots_1.values[i])
# print(one_hots_1.keys())
# for key in one_hots_1.keys():
#     if not key in invalids:
#         print(one_hots_1[key])
# value_arrays = np.array(list(one_hots_1.values()))
# value_tensors = torch.from_numpy(value_arrays)
# output_targets = torch.argmax(outputs,1)
# value_targets = torch.argmax(value_tensors,1)
# acc = torch.sum(output_targets == value_targets)
# print('accuracy',acc*100/all_outputs.shape[0])
f = open('valids.txt','w')
for valid in valids:
    print(valid,file=f)
g = open('all_outputs.txt','w')
for output in all_outputs:
    print(output,file=g)