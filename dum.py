import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
one_hots_1 = onehot('dataset_TMA/tma_info/ZT80_gleason_scores.csv','patho_2')
outputs = torch.load('all_outputs_2.pt').cuda()
targets = torch.from_numpy(np.array(list(one_hots_1.values()))).cuda()

outputs = F.softmax(outputs,dim=1)
correct = 0
for i in range(outputs.shape[0]):
    if torch.argmax(outputs[i]) == torch.argmax(targets[i]):
        print(torch.argmax(outputs[i]),i)
        correct+=1
print(correct)