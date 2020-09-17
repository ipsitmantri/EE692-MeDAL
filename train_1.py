import torch
import torch.nn as nn
from torchvision import datasets,models,transforms
import os
import time
import copy
import torch.optim as optim

if __name__ == '__main__':

    img_size = 224
    data_path = '..\\harvard_data\\train_val_split_data'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    data_aug = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        ]),
    }

    # The images are stored in a dedicated folder for each class in data_path/train and data_path/val
    # images['train'] contains tuples of image and label automatically given (scalar) for each class
    # where the images for class 'ant' are stored in data_path/train/ant
    images = {x:datasets.ImageFolder(os.path.join(data_path,x),data_aug[x])
              for x in ['train','val']}
    # DataLoader divides these into shuffled batches
    dataloaders = {x:torch.utils.data.DataLoader(images[x],batch_size=4,shuffle=True,num_workers=4)
                   for x in ['train','val']}
    print(images['train'].classes)
    data_sizes = {x:len(images[x])
                  for x in ['train','val']}


    def train_model(model,criterion,optimiser,scheduler,epochs):
        best_val_model = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        stats = {
            'train': {
                'loss':[],
                'acc':[]
            },
            'val': {
                'loss':[],
                'acc':[]
            }
        }
        since = time.time()
        for epoch in range(epochs):
            print(f'Epoch: {epoch+1}/{epochs}')
            print('-'*10)
            for phase in ['train','val']:
                if phase=='train':
                    model.train()
                else :
                    model.eval()
                running_loss = 0.0
                running_strikes = 0
                for _X,_y in dataloaders[phase]:
                    _X,_y = _X.to(device),_y.to(device)
                    optimiser.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                        # bs, ncrops, c, h, w = _X.size()
                        outputs = model(_X)
                        # outputs = outputs.view(bs,ncrops,-1).mean(1)
                        preds = torch.argmax(outputs,dim=1)
                        _ground_truths = _y.data
                        loss = criterion(outputs,_ground_truths)
                        running_loss += loss.item()*(_X.shape[0])
                        running_strikes += torch.sum(preds==_ground_truths)

                        if phase == 'train':
                            loss.backward()
                            optimiser.step()

                epoch_loss = running_loss/data_sizes[phase]
                epoch_acc = running_strikes*1.0/data_sizes[phase]
                stats[phase]['loss'].append(epoch_loss)
                stats[phase]['acc'].append(epoch_acc)

                if phase == 'val':
                    scheduler.step(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = stats[phase]['acc'][-1]
                        best_val_model = copy.deepcopy(model.state_dict())
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            print('-'*40)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        model.load_state_dict(best_val_model)
        return model,stats

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(num_ftrs,256),nn.ReLU(),nn.Dropout(0.4),nn.Linear(256,4))
    model.load_state_dict(torch.load('logs\\train_2_resnet18_lr=0.001_frozen.pt'))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(),lr=0.001,momentum=0.9,nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser,patience=4,mode='min',factor=0.2)
    model, stats = train_model(model,criterion,optimiser,scheduler,40)

    f = open('logs\\train_2_resnet18_lr=0.001_frozen.log','w')
    for i in range(40):
        tl = stats['train']['loss'][i]
        ta = stats['train']['acc'][i]
        vl = stats['val']['loss'][i]
        va = stats['val']['acc'][i]
        f.write(f'{i},{tl},{ta},{vl},{va}\n')
    f.close()
    torch.save(model.state_dict(), 'logs\\train_2_resnet18_lr=0.001_frozen.pt')
