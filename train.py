import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, self.L * self.K),nn.ReLU())
        self.pre_final = nn.Sequential(nn.Linear(self.L * self.K, self.L * self.K),nn.ReLU())
        self.final = nn.Sequential(nn.Linear(self.L * self.K,10),nn.Softmax(dim=1))
        # self.classifier_debug = nn.Sequential(nn.Linear(self.L, 10), nn.Softmax(dim=1))

    def forward(self, x):
        H = self.pre_trained_net(x)
        A_V = self.l1(H)
        A_U = self.l2(H)
        A = self.l3(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        Y = self.classifier(M)
        Y = self.pre_final(Y)
        Y = self.final(Y)
        return Y

def preprocess(image):
    transform = transforms.Compose([
        # transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.RandomCrop([224, 224]),
        # transforms.Resize(224,interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    return transform(image)


# encoding: [0+0,3+3,4+4,5+5,3+4,3+5,4+3,4+5,5+3,5+4]

def onehot(file):
    filew = pd.read_csv(file)
    # one_hots = np.empty((filew.shape[0], 11),dtype=object)
    one_hots = {}
    for i in range(filew.shape[0]):
        primary_score = filew.loc[i][0][-3]
        secondary_score = filew.loc[i][0][-1]
        # one_hots[i,0] = filew.loc[i][0][:-4]
        key = filew.loc[i][0][:-4]
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


PARENT_DIR = 'dataset_TMA'
train_val_masks_dir = os.path.join(PARENT_DIR, 'train_validation_patches_750')
train_dirs = []
val_dirs = []
train_dirs_net = []
val_dirs_net = []
for dir in os.listdir(train_val_masks_dir):
    if (dir.startswith('ZT111') or dir.startswith('ZT199') or dir.startswith('ZT204')):
        train_dirs.append(dir)
    elif dir.startswith('ZT76'):
        val_dirs.append(dir)
    if len(os.listdir(os.path.join(train_val_masks_dir,dir))) != 0:
        if (dir.startswith('ZT111') or dir.startswith('ZT199') or dir.startswith('ZT204')):
            train_dirs_net.append(dir)
        elif dir.startswith('ZT76'):
            val_dirs_net.append(dir)

tma_info = os.path.join(PARENT_DIR, 'tma_info')
train_arrays = ['ZT111', 'ZT199', 'ZT204']
val_arrays = ['ZT76']
one_hots_train = {}
one_hots_val = {}

for file in os.listdir(tma_info):
    if file[0:5] in train_arrays:
        one_hots_train[file[0:5]] = onehot(os.path.join(tma_info, file))
    if file[0:4] in val_arrays:
        one_hots_val[file[0:4]] = onehot(os.path.join(tma_info,file))

preTrainedModel = models.resnet18(pretrained=True)

for param in preTrainedModel.parameters():
    param.requires_grad = False

attentionModel = Attention(preTrainedModel)
attentionModel = attentionModel.cuda()


optimizer = optim.SGD(attentionModel.parameters(),lr=1e-4,momentum=0.9,nesterov=True)
lossFunction = nn.NLLLoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=4,mode='min',factor=0.2,verbose=True)
writer = SummaryWriter()

loss_train = 0
loss_val = 0
all_outputs_train = {}
all_outputs_val = {}
running_loss_train = 0
running_loss_val = 0
empty_train = []
empty_val = []
for idx,dir in enumerate(train_dirs):
    paths = os.path.join(train_val_masks_dir,dir)
    if len(os.listdir(paths)) == 0:
        empty_train.append(paths.split('\\')[2])
for idx,dir in enumerate(val_dirs):
    paths = os.path.join(train_val_masks_dir,dir)
    if len(os.listdir(paths)) == 0:
        empty_val.append(paths.split('\\')[2])
one_hots_val_processed = {}
one_hots_train_processed = {}
for array in train_arrays:
    dictionary = one_hots_train[array]
    one_hots_train_processed[array] = {}
    for key in dictionary.keys():
        if key not in empty_train:
            one_hots_train_processed[array][key] = dictionary[key]
for array in val_arrays:
    dictionary = one_hots_val[array]
    one_hots_val_processed[array] = {}
    for key in dictionary.keys():
        if key not in empty_val:
            one_hots_val_processed[array][key] = dictionary[key]
def trainLoop(attentionModel,optimizer,lossFunction,scheduler,writer,EPOCHS):
    for epoch in range(EPOCHS):
        running_loss_train = 0
        running_loss_val = 0
        print("Epoch: {}/{}".format(epoch,EPOCHS-1))
        for idx,dir in tqdm(enumerate(train_dirs_net)):
            attentionModel.train()
            paths = os.path.join(train_val_masks_dir,dir)
            bag = np.zeros((len(os.listdir(paths)),3,224,224),dtype=np.uint8)
            Y = one_hots_train_processed[dir[0:5]][dir]
            for i in range(len(os.listdir(paths))):
                path = os.listdir(paths)[i]
                img = Image.open(os.path.join(paths,path))
                img = img.convert('RGB')
                img = preprocess(img)
                bag[i,:,:,:] = img
            bag = torch.from_numpy(bag).type(torch.float32)
            bag = bag.cuda()
            Y = torch.from_numpy(Y).type(torch.LongTensor).view(1,10).cuda()
            _,target = torch.max(Y,1)
            target = target.cuda()
            optimizer.zero_grad()
            outputs = attentionModel(bag)
            loss_train = lossFunction(outputs,target)
            loss_train.backward()
            optimizer.step()
            all_outputs_train[dir] = outputs.cpu()
            running_loss_train += loss_train.item()
        writer.add_scalar('Train Loss',running_loss_train/len(train_dirs),epoch)
        print("Train Loss: {}".format(running_loss_train/len(train_dirs)))


        with torch.no_grad():
            attentionModel.eval()
            for idx,dir in tqdm(enumerate(val_dirs_net)):
                paths = os.path.join(train_val_masks_dir, dir)
                bag = np.zeros((len(os.listdir(paths)),3,224,224), dtype=np.uint8)
                Y_val = one_hots_val_processed[dir[0:4]][dir]
                for i in range(len(os.listdir(paths))):
                    path = os.listdir(paths)[i]
                    img = Image.open(os.path.join(paths, path))
                    img = img.convert('RGB')
                    img = preprocess(img)
                    bag[i, :, :, :] = img
                bag = torch.from_numpy(bag).type(torch.float32)
                bag = bag.cuda()
                Y_val = torch.from_numpy(Y_val).type(torch.LongTensor).view(1, 10).cuda()
                _, target = torch.max(Y_val, 1)
                target = target.cuda()
                outputs_val = attentionModel(bag)
                all_outputs_val[dir] = outputs_val.cpu()
                loss_val = lossFunction(outputs_val, target)
                running_loss_val += loss_val.item()
            writer.add_scalar('Validation Loss', running_loss_val/len(val_dirs), epoch)
            scheduler.step(running_loss_val/len(val_dirs))
        print("Val Loss: {}".format(running_loss_val/len(val_dirs)))
    return attentionModel,all_outputs_train,all_outputs_val

EPOCHS = int(input("Enter the number of epochs: "))
model,all_outputs_train,all_outputs_val = trainLoop(attentionModel, optimizer, lossFunction, scheduler, writer, EPOCHS)
torch.save(model.state_dict(),'trained_model.pth')
import pickle
f = open('all_outputs_train.pkl','wb')
pickle.dump(all_outputs_train,f)
f.close()
g = open('all_outputs_val.pkl','wb')
pickle.dump(all_outputs_val,g)
g.close()