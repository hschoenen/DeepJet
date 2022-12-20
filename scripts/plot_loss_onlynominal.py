import numpy as np
import torch
print("finish import")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_names = ['nominal_bs2000']
tagger = 'DF_Run2'
dirz = [f'/eos/user/h/heschone/DeepJet/Train_{tagger}/{model_name}/' \
        for model_name in model_names]

nominal_epochs = 39
#adversarial_epochsA = 73
#adversarial_epochsB = 78
#adversarial_epochsC = 78

paths = {
    'nominal_bs2000' : [dirz[0] + f'checkpoint_epoch_{i}.pth' for i in range(1,nominal_epochs+1)],
    #'adversarialA' : [dirz[1] + f'checkpoint_epoch_{i}.pth' for i in range(1,adversarial_epochsA+1)],
    #'adversarialB' : [dirz[2] + f'checkpoint_epoch_{i}.pth' for i in range(1,adversarial_epochsB+1)],
    #'adversarialC' : [dirz[3] + f'checkpoint_epoch_{i}.pth' for i in range(1,adversarial_epochsC+1)]
    }

for p,key in enumerate(paths):
    
    tr_losses = []
    val_losses = []
        
    for i in range(len(paths[key])):
        checkpoint = torch.load(paths[key][i], map_location=torch.device(device))
        tr_losses.append(checkpoint['train_loss'])
        val_losses.append(checkpoint['val_loss'])
    
    np.save(dirz[p] + f'loss_trainval.npy',np.array([tr_losses,val_losses]))
