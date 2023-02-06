import numpy as np
import torch
print("finish import")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_names = ['nominal','adversarial']#,'adversarial_flavour_006008010','adversarial_flavour_008009010']
#,'adversarial_flavour_010009010','adversarial','adversarial_flavour']
tagger = 'DF_Run2' #'DF'
dirz = {
    'nominal': f'/eos/user/h/heschone/DeepJet/Train_DF_Run2/nominal/',
    'adversarial': f'/eos/user/h/heschone/DeepJet/Train_DF_Run2/fgsm/',
    'adversarial_flavour_006008010': f'/eos/user/h/heschone/DeepJet/Train_DF_Run2/adversarial_flavour_006008010/',
    'adversarial_flavour_008009010': f'/eos/user/h/heschone/DeepJet/Train_DF_Run2/adversarial_flavour_008009010/',
    'adversarial_flavour_010009010': f'/eos/user/h/heschone/DeepJet/Train_DF_Run2/adversarial_flavour_010009010/',
}

nominal_epochs = 39
adversarial_epochs = 78
adversarial_flavour_epochs = 62

paths = {
    'nominal' : [dirz['nominal'] + f'checkpoint_epoch_{i}.pth' for i in range(1,nominal_epochs+1)],
    'adversarial' : [dirz['adversarial'] + f'checkpoint_epoch_{i}.pth' for i in range(1,adversarial_epochs+1)],
    'adversarial_flavour_006008010' : [dirz['adversarial_flavour_006008010'] + f'checkpoint_epoch_{i}.pth' for i in range(1,adversarial_flavour_epochs+1)],
    'adversarial_flavour_008009010' : [dirz['adversarial_flavour_008009010'] + f'checkpoint_epoch_{i}.pth' for i in range(1,adversarial_epochs+1)],
    'adversarial_flavour_010009010' : [dirz['adversarial_flavour_010009010'] + f'checkpoint_epoch_{i}.pth' for i in range(1,adversarial_epochs+1)],
    }

for model in model_names:
    
    tr_losses = []
    val_losses = []
        
    for i in range(len(paths[model])):
        checkpoint = torch.load(paths[model][i], map_location=torch.device(device))
        tr_losses.append(checkpoint['train_loss'])
        val_losses.append(checkpoint['val_loss'])
    
    np.save(dirz[model] + f'loss_trainval.npy',np.array([tr_losses,val_losses]))
