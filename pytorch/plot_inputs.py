# Hendrik Schönen
# This script is supposed to get the training data and save the input features before and after applying an attack
# It has to be executed like: python3 pytorch/plot_inputs.py

# specify number of jets, that should be saved
batchsize = 40000
# specify variables, that should be saved
variables = ['jet_pt', 'jet_eta']
# specify model
model_name = 'fgsm-0_01'
modelDir = '/net/scratch_cms3a/hschoenen/deepjet/results/' + model_name
modelFile = 'checkpoint_best_loss.pth'
# specify input file
inputFile = '/net/scratch_cms3a/hschoenen/deepjet/data/testData/dataCollection.djcdc'

import torch 
import torch.nn as nn
from pytorch_first_try import training_base
from pytorch_deepjet_run2 import *
import os
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
import numpy as np
from attacks import *
from definitions import epsilons_per_feature, vars_per_candidate
from definitions import *
print('imports finished')

# dictionary for the
variable_indices = {
    # global
    'jet_pt': ['glob',0],
    'jet_eta': ['glob',1],
    'nCpfcand': ['glob',2],
    'nNpfcand': ['glob',3],
    # cpf
    'Cpfcan_BtagPf_trackDeltaR': ['cpf',0,3],
    'Cpfcan_BtagPf_trackPParRatio': ['cpf',0,4],
    'Cpfcan_BtagPf_trackSip2dVal': ['cpf',0,5],
    'Cpfcan_BtagPf_trackSip2dSig': ['cpf',0,6],
    'Cpfcan_BtagPf_trackSip3dVal': ['cpf',0,7],
    'Cpfcan_BtagPf_trackSip3dSig': ['cpf',0,8],
    'Cpfcan_BtagPf_trackJetDistVal': ['cpf',0,9],
    # npf
    'Npfcan_ptrel': ['npf',0,0],
    'Npfcan_deltaR': ['npf',0,1],
    # vtx
    'sv_pt': ['vtx',0,0],
    'sv_deltaR': ['vtx',0,1],
    'sv_mass': ['vtx',0,2],
    'sv_dxy': ['vtx',0,6],
    'sv_dxy': ['vtx',0,7],
    'sv_d3d': ['vtx',0,8],
    'sv_d3dsig': ['vtx',0,9]
}

def save_features(inputs, targets, outputs, attack=""):
    # create saving directory
    if not os.path.isdir('{}/variables'.format(modelDir)):
        os.mkdir('{}/variables'.format(modelDir))
    if not os.path.isdir('{}/variables/{}'.format(modelDir,attack)):
        os.mkdir('{}/variables/{}'.format(modelDir,attack))
    # save inputs
    x_glob, x_cpf, x_npf, x_vtx = inputs[0], inputs[1], inputs[2], inputs[3]
    for variable in variables:
        variable_index = variable_indices[variable]
        if variable_index[0]=='glob':
            variable_values = x_glob[:,variable_index[1]].cpu().clone().detach().numpy()
        elif variable_index[0]=='cpf':
            variable_values = x_cpf[:,variable_index[1],variable_index[2]].cpu().clone().detach().numpy()
        elif variable_index[0]=='npf':
            variable_values = x_npf[:,variable_index[1],variable_index[2]].cpu().clone().detach().numpy()
        elif variable_index[0]=='vtx':
            variable_values = x_vtx[:,variable_index[1],variable_index[2]].cpu().clone().detach().numpy()
        else:
            raise Exception('variable branch does not exist!')
        np.save('{}/variables/{}/{}.npy'.format(modelDir,attack,variable), variable_values)
    # save targets
    target_names = ['isB','isBB','isLeptonicB','isC','isUDS','isG']
    for i,target_name in enumerate(target_names):
        variable_values = targets[:,i].cpu().clone().detach().numpy()
        np.save('{}/variables/{}/{}.npy'.format(modelDir,attack,target_name), variable_values)
    # save predictions
    prediction_names = ['prob_isB','prob_isBB','prob_isLeptonicB','prob_isC','prob_isUDS','prob_isG']
    for i,prediction_name in enumerate(prediction_names):
        variable_values = outputs[:,i].cpu().clone().detach().numpy()
        np.save('{}/variables/{}/{}.npy'.format(modelDir,attack,prediction_name), variable_values)

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

model = DeepJet_Run2(num_classes = 6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# build training base (from pytorch/pytorch_first_try.py)
train=training_base(model = model, criterion = cross_entropy_one_hot, optimizer = None, scheduler = None, evaluation_inputFile=inputFile, splittrainandtest=0)
train.train_data.maxFilesOpen=1
print("training_base created")

# build train_generator
train.train_data.setBatchSize(batchsize)
traingen = train.train_data.invokeGenerator()
traingen.setBatchSize(batchsize)
traingen.prepareNextEpoch()
train_generator = traingen.feedNumpyData()
print("train_generator created")

# load model
check = torch.load(modelDir+'/'+modelFile, map_location=torch.device('cpu'))
model.load_state_dict(check['state_dict'])
model.to(device)
model.train()

# Processing the data in train_loop()
features_list, truth_list = next(train_generator)
glob = torch.Tensor(features_list[0]).to(device)
cpf = torch.Tensor(features_list[1]).to(device)
npf = torch.Tensor(features_list[2]).to(device)
vtx = torch.Tensor(features_list[3]).to(device)
y = torch.Tensor(truth_list[0]).to(device)
pred = (model(glob,cpf,npf,vtx)).detach()
save_features((glob,cpf,npf,vtx),y,pred,attack="nominal")

# load epsilon factors for adversarial attacks
epsilon_factors = {
    'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(device),
    'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(device),
    'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(device),
    'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(device),
    }

# FGSM epsilon=0.01
x_glob, x_cpf, x_npf, x_vtx = fgsm_attack(sample=(glob,cpf,npf,vtx), epsilon=0.01, dev=device, targets=y, thismodel=model, thiscriterion=train.criterion, restrict_impact=0.2, epsilon_factors=epsilon_factors, allow_zeros=True)
pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,attack="fgsm-0_01")
    

    