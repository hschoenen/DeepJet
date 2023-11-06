# Hendrik Schönen
# This script is supposed to get the training data and save arrays for plotting the loss surface
# It has to be executed like: python3 pytorch/plot_loss_surface.py

# specify which loss surfaces should be computed [variable_x, variable_y, delta_x, delta_y, Nsteps]
surfaces = [['jet_pt','jet_eta',0.1,0.01,100],['Cpfcan_BtagPf_trackSip3dVal','sv_d3d',0.0005,0.0005,100]]
# specify which jets should be saved
save_jets = ['isB','isC','isUDS']
N_save = 10

# specify model
model_name = 'nominal'
modelDir = '/net/data_cms/institut_3a/hschoenen/models/' + model_name
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
    'nsv': ['glob',4],
    'npv': ['glob',5],
    'TagVarCSV_trackSumJetEtRatio': ['glob',6],
    'TagVarCSV_trackSumJetDeltaR': ['glob',7],
    'TagVarCSV_vertexCategory': ['glob',8],
    'TagVarCSV_trackSip2dValAboveCharm': ['glob',9],
    'TagVarCSV_trackSip2dSigAboveCharm': ['glob',10],
    'TagVarCSV_trackSip3dValAboveCharm': ['glob',11],
    'TagVarCSV_trackSip3dSigAboveCharm': ['glob',12],
    'TagVarCSV_jetNSelectedTracks': ['glob',13],
    'TagVarCSV_jetNTracksEtaRel': ['glob',14],
    # cpf
    'Cpfcan_BtagPf_trackEtaRel': ['cpf',0,0],
    'Cpfcan_BtagPf_trackPtRel': ['cpf',0,1],
    'Cpfcan_BtagPf_trackPPar': ['cpf',0,2],
    'Cpfcan_BtagPf_trackDeltaR': ['cpf',0,3],
    'Cpfcan_BtagPf_trackPParRatio': ['cpf',0,4],
    'Cpfcan_BtagPf_trackSip2dVal': ['cpf',0,5],
    'Cpfcan_BtagPf_trackSip2dSig': ['cpf',0,6],
    'Cpfcan_BtagPf_trackSip3dVal': ['cpf',0,7],
    'Cpfcan_BtagPf_trackSip3dSig': ['cpf',0,8],
    'Cpfcan_BtagPf_trackJetDistVal': ['cpf',0,9],
    'Cpfcan_ptrel': ['cpf',0,10],
    'Cpfcan_drminsv': ['cpf',0,11],
    'Cpfcan_VTX_ass': ['cpf',0,12],
    'Cpfcan_puppiw': ['cpf',0,13],
    'Cpfcan_chi2': ['cpf',0,14],
    'Cpfcan_quality': ['cpf',0,15],
    # npf
    'Npfcan_ptrel': ['npf',0,0], 
    'Npfcan_deltaR': ['npf',0,1],
    'Npfcan_isGamma': ['npf',0,2], 
    'Npfcan_HadFrac': ['npf',0,3], 
    'Npfcan_drminsv': ['npf',0,4], 
    'Npfcan_puppiw': ['npf',0,5],
    # vtx
    'sv_pt': ['vtx',0,0],
    'sv_deltaR': ['vtx',0,1],
    'sv_mass': ['vtx',0,2],
    'sv_ntracks': ['vtx',0,3],
    'sv_chi2': ['vtx',0,4],
    'sv_normchi2': ['vtx',0,5],
    'sv_dxy': ['vtx',0,6],
    'sv_dxysig': ['vtx',0,7],
    'sv_d3d': ['vtx',0,8],
    'sv_d3dsig': ['vtx',0,9],
    'sv_costhetasvpv': ['vtx',0,10],
    'sv_enratio': ['vtx',0,11],
}

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
batchsize = 40000
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
model.eval()

# Processing the data in train_loop()
features_list, truth_list = next(train_generator)
glob = torch.Tensor(features_list[0]).to(device)
cpf = torch.Tensor(features_list[1]).to(device)
npf = torch.Tensor(features_list[2]).to(device)
vtx = torch.Tensor(features_list[3]).to(device)
y = torch.Tensor(truth_list[0]).to(device)
pred = (model(glob,cpf,npf,vtx)).detach()
pred_softmax = nn.Softmax(dim=1)(model(glob,cpf,npf,vtx)).cpu().detach().numpy()

# function to get the values of a specific variable
def get_variable(inputs, variable):
    x_glob, x_cpf, x_npf, x_vtx = inputs[0], inputs[1], inputs[2], inputs[3]
    variable_index = variable_indices[variable]
    if variable_index[0]=='glob':
        values = x_glob[:,variable_index[1]].cpu().clone().detach().numpy()
    elif variable_index[0]=='cpf':
        values = x_cpf[:,variable_index[1],variable_index[2]].cpu().clone().detach().numpy()
    elif variable_index[0]=='npf':
        values = x_npf[:,variable_index[1],variable_index[2]].cpu().clone().detach().numpy()
    elif variable_index[0]=='vtx':
        values = x_vtx[:,variable_index[1],variable_index[2]].cpu().clone().detach().numpy()
    else:
        raise Exception('variable branch does not exist!')
    return values

# funtion to change an input value of a specific jet
def set_variable(inputs, variable, index, value):
    x_glob, x_cpf, x_npf, x_vtx = inputs[0], inputs[1], inputs[2], inputs[3]
    variable_index = variable_indices[variable]
    if variable_index[0]=='glob':
        print(index, ' ', value)
        x_glob[index,variable_index[1]] = value
    elif variable_index[0]=='cpf':
        x_cpf[index,variable_index[1],variable_index[2]] = value
    elif variable_index[0]=='npf':
        x_npf[index,variable_index[1],variable_index[2]] = value
    elif variable_index[0]=='vtx':
        x_vtx[index,variable_index[1],variable_index[2]] = value
    else:
        raise Exception('variable branch does not exist!')
    return x_glob,x_cpf,x_npf,x_vtx
    
# function to save the loss surface
def save_loss_surface(inputs, surface, jet_index, name):
    print('jet name :',name)
    print('surface :',surface)
    # create saving directory
    if not os.path.isdir('{}/loss_surfaces'.format(modelDir)):
        os.mkdir('{}/loss_surfaces'.format(modelDir))
    if not os.path.isdir('{}/loss_surfaces/{}'.format(modelDir,name)):
        os.mkdir('{}/loss_surfaces/{}'.format(modelDir,name))
    # get information about the surface
    variable_x = surface[0]
    variable_y = surface[1]
    delta_x = surface[2]
    delta_y = surface[3]
    Nsteps = surface[4]
    # get the chosen jet
    glob, cpf, npf, vtx = inputs[0], inputs[1], inputs[2], inputs[3]
    value_x = get_variable((glob,cpf,npf,vtx), variable_x)[jet_index]
    value_y = get_variable((glob,cpf,npf,vtx), variable_y)[jet_index]
    print('x variable value: ',value_x)
    print('y variable value: ',value_y)
    jet_glob = glob[jet_index,:].clone().detach()
    jet_cpf = cpf[jet_index,:,:].clone().detach()
    jet_npf = npf[jet_index,:,:].clone().detach()
    jet_vtx = vtx[jet_index,:,:].clone().detach()
    N = (2*Nsteps+1)**2
    jet_glob = torch.unsqueeze(jet_glob,0)
    jet_glob = jet_glob.repeat(N,1)
    jet_cpf = torch.unsqueeze(jet_cpf,0)
    jet_cpf = jet_cpf.repeat(N,1,1)
    jet_npf = torch.unsqueeze(jet_npf,0)
    jet_npf = jet_npf.repeat(N,1,1)
    jet_vtx = torch.unsqueeze(jet_vtx,0)
    jet_vtx = jet_vtx.repeat(N,1,1)
    
    n = 0
    for index_x in range (-Nsteps,Nsteps+1):
        for index_y in range (-Nsteps,Nsteps+1):
            # change the value of x
            variable_index = variable_indices[variable_x]
            if variable_index[0]=='glob':
                jet_glob[n,variable_index[1]] = value_x+index_x*delta_x
            elif variable_index[0]=='cpf':
                jet_cpf[n,variable_index[1],variable_index[2]] = value_x+index_x*delta_x
            elif variable_index[0]=='npf':
                jet_npf[n,variable_index[1],variable_index[2]] = value_x+index_x*delta_x
            elif variable_index[0]=='vtx':
                jet_vtx[n,variable_index[1],variable_index[2]] = value_x+index_x*delta_x
            # change the value of y
            variable_index = variable_indices[variable_y]
            if variable_index[0]=='glob':
                jet_glob[n,variable_index[1]] = value_y+index_y*delta_y
            elif variable_index[0]=='cpf':
                jet_cpf[n,variable_index[1],variable_index[2]] = value_y+index_y*delta_y
            elif variable_index[0]=='npf':
                jet_npf[n,variable_index[1],variable_index[2]] = value_y+index_y*delta_y
            elif variable_index[0]=='vtx':
                jet_vtx[n,variable_index[1],variable_index[2]] = value_y+index_y*delta_y
            n += 1
    # evaluate the model        
    pred =  nn.Softmax(dim=1)((model(jet_glob,jet_cpf,jet_npf,jet_vtx))).detach()
    x_array = get_variable((jet_glob,jet_cpf,jet_npf,jet_vtx), variable_x)
    y_array = get_variable((jet_glob,jet_cpf,jet_npf,jet_vtx), variable_y)
    print('model predictions: ',pred)
    print('x variable array: ',x_array)
    print('y variable array: ',y_array)
    # save the loss surface
    np.save('{}/loss_surfaces/{}/{}-{}-x.npy'.format(modelDir,name,variable_x,variable_y), x_array)
    np.save('{}/loss_surfaces/{}/{}-{}-y.npy'.format(modelDir,name,variable_x,variable_y), y_array)
    np.save('{}/loss_surfaces/{}/{}-{}-prediction.npy'.format(modelDir,name,variable_x,variable_y), pred.cpu().numpy())
    print('{}/loss_surfaces/{}/{}-{}-prediction.npy saved'.format(modelDir,name,variable_x,variable_y))

n_isB = 0
n_isC = 0
n_isUDS = 0
for jet_index in range(batchsize):
    isB = y[jet_index,0]
    isC = y[jet_index,3]
    isUDS = y[jet_index,4]
    prob_isB = pred_softmax[jet_index,0]
    prob_isC = pred_softmax[jet_index,3]
    prob_isUDS = pred_softmax[jet_index,4]
    # save b jets
    if isB==1 and n_isB<N_save:
        for surface in surfaces:
            save_loss_surface((glob,cpf,npf,vtx), surface, jet_index, name='bjet{}'.format(n_isB+1))
        n_isB += 1
    # save c jets
    if isC==1 and n_isC<N_save:
        for surface in surfaces:
            save_loss_surface((glob,cpf,npf,vtx), surface, jet_index, name='cjet{}'.format(n_isC+1))
        n_isC += 1
    # save uds jets
    if isUDS==1 and n_isUDS<N_save:
        for surface in surfaces:
            save_loss_surface((glob,cpf,npf,vtx), surface, jet_index, name='udsjet{}'.format(n_isUDS+1))
        n_isUDS += 1
    