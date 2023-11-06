# Hendrik Schönen
# This script is supposed to get the training data and save the input features before and after applying an attack
# It has to be executed like: python3 pytorch/plot_inputs.py

# specify number of jets, that should be saved
batchsize = 400000
# specify variables, that should be saved
variables = ['jet_pt', 'jet_eta', 'nCpfcand','nNpfcand', 'nsv','npv', 'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 'TagVarCSV_jetNTracksEtaRel',
             'Cpfcan_BtagPf_trackEtaRel', 'Cpfcan_BtagPf_trackPtRel', 'Cpfcan_BtagPf_trackPPar', 'Cpfcan_BtagPf_trackDeltaR', 'Cpfcan_BtagPf_trackPParRatio', 'Cpfcan_BtagPf_trackSip2dVal', 'Cpfcan_BtagPf_trackSip2dSig', 'Cpfcan_BtagPf_trackSip3dVal',
'Cpfcan_BtagPf_trackSip3dSig', 'Cpfcan_BtagPf_trackJetDistVal', 'Cpfcan_ptrel', 'Cpfcan_drminsv', 'Cpfcan_VTX_ass', 'Cpfcan_puppiw', 'Cpfcan_chi2', 'Cpfcan_quality',
             'Npfcan_ptrel', 'Npfcan_deltaR', 'Npfcan_isGamma', 'Npfcan_HadFrac', 'Npfcan_drminsv', 'Npfcan_puppiw',
             'sv_pt','sv_deltaR', 'sv_mass', 'sv_ntracks', 'sv_chi2', 'sv_normchi2', 'sv_dxy', 'sv_dxysig', 'sv_d3d', 'sv_d3dsig', 'sv_costhetasvpv', 'sv_enratio']

# specify model
model_name = 'fgsm-0_1'
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
        print(x_glob.size())
        print(len(variable_values))
        np.save('{}/variables/{}/{}.npy'.format(modelDir,attack,variable), variable_values)
        print('{}/variables/{}/{}.npy saved'.format(modelDir,attack,variable))
    # save targets
    target_names = ['isB','isBB','isLeptonicB','isC','isUDS','isG']
    for i,target_name in enumerate(target_names):
        variable_values = targets[:,i].cpu().clone().detach().numpy()
        np.save('{}/variables/{}/{}.npy'.format(modelDir,attack,target_name), variable_values)
        print('{}/variables/{}/{}.npy saved'.format(modelDir,attack,target_name))
    # save predictions
    prediction_names = ['prob_isB','prob_isBB','prob_isLeptonicB','prob_isC','prob_isUDS','prob_isG']
    softmax_outputs = nn.Softmax(dim=1)(outputs).cpu().detach().numpy()
    for i,prediction_name in enumerate(prediction_names):
        variable_values = outputs[:,i].cpu().clone().detach().numpy()
        np.save('{}/variables/{}/{}.npy'.format(modelDir,attack,prediction_name), variable_values)
        print('{}/variables/{}/{}.npy saved'.format(modelDir,attack,prediction_name))
        variable_values = softmax_outputs[:,i]
        np.save('{}/variables/{}/softmax_{}.npy'.format(modelDir,attack,prediction_name), variable_values)
        print('{}/variables/{}/softmax_{}.npy saved'.format(modelDir,attack,prediction_name))

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

# load model and set it to evaluation mode
check = torch.load(modelDir+'/'+modelFile, map_location=torch.device('cpu'))
model.load_state_dict(check['state_dict'])
model.to(device)
model.eval()

# save inputs and outputs
features_list, truth_list = next(train_generator)
glob = torch.Tensor(features_list[0]).to(device)
cpf = torch.Tensor(features_list[1]).to(device)
npf = torch.Tensor(features_list[2]).to(device)
vtx = torch.Tensor(features_list[3]).to(device)
y = torch.Tensor(truth_list[0]).to(device)
pred = (model(glob,cpf,npf,vtx)).detach()
save_features((glob,cpf,npf,vtx),y,pred,attack="nominal")

'''
# save all model inputs
x_glob = glob.clone().detach()
x_cpf = cpf.clone().detach()
x_npf = npf.clone().detach()
x_vtx = vtx.clone().detach()
glob_array = x_glob.cpu().clone().detach().numpy()
cpf_array = x_cpf.cpu().clone().detach().numpy()
npf_array = x_npf.cpu().clone().detach().numpy()
vtx_array = x_vtx.cpu().clone().detach().numpy()
np.save('{}/variables/glob_array.npy'.format(modelDir), glob_array)
np.save('{}/variables/cpf_array.npy'.format(modelDir), cpf_array)
np.save('{}/variables/npf_array.npy'.format(modelDir), npf_array)
np.save('{}/variables/vtx_array.npy'.format(modelDir), vtx_array)


# save all model gradients
model.train()
x_glob = glob.clone().detach()
x_cpf = cpf.clone().detach()
x_npf = npf.clone().detach()
x_vtx = vtx.clone().detach()
x_glob.requires_grad = True
x_cpf.requires_grad = True
x_npf.requires_grad = True
x_vtx.requires_grad = True
pred = model(x_glob,x_cpf,x_npf,x_vtx)
loss = cross_entropy_one_hot(pred, y)
model.zero_grad()
loss.backward()
with torch.no_grad():
    dx_glob = x_glob.grad.detach()
    dx_cpf = x_cpf.grad.detach()
    dx_npf = x_npf.grad.detach()
    dx_vtx = x_vtx.grad.detach()
glob_gradients = dx_glob.cpu().clone().detach().numpy()
cpf_gradients = dx_cpf.cpu().clone().detach().numpy()
npf_gradients = dx_npf.cpu().clone().detach().numpy()
vtx_gradients = dx_vtx.cpu().clone().detach().numpy()
np.save('{}/variables/glob_gradients.npy'.format(modelDir), glob_gradients)
np.save('{}/variables/cpf_gradients.npy'.format(modelDir), cpf_gradients)
np.save('{}/variables/npf_gradients.npy'.format(modelDir), npf_gradients)
np.save('{}/variables/vtx_gradients.npy'.format(modelDir), vtx_gradients)
model.eval()
'''

# load epsilon factors for adversarial attacks
epsilon_factors = {
    'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(device),
    'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(device),
    'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(device),
    'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(device),
    }

'''
# FGSM epsilon=0.05
model.train()
x_glob, x_cpf, x_npf, x_vtx = fgsm_attack(sample=(glob,cpf,npf,vtx), epsilon=0.05, dev=device, targets=y, thismodel=model, thiscriterion=train.criterion, restrict_impact=0.2, epsilon_factors=epsilon_factors, allow_zeros=True)
model.eval()
pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,attack="fgsm-0_05")

# FGSM epsilon=0.1
model.train()
x_glob, x_cpf, x_npf, x_vtx = fgsm_attack(sample=(glob,cpf,npf,vtx), epsilon=0.1, dev=device, targets=y, thismodel=model, thiscriterion=train.criterion, restrict_impact=0.2, epsilon_factors=epsilon_factors, allow_zeros=True)
model.eval()
pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,attack="fgsm-0_1")

# Gaussian noise with epsilon=0.1
model.train()
x_glob, x_cpf, x_npf, x_vtx = gaussian_noise(sample=(glob,cpf,npf,vtx), epsilon=1, restrict_impact=-1, epsilon_factors=epsilon_factors)
model.eval()
pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,attack="gaussian-1")

# Gaussian noise with epsilon=1
model.train()
x_glob, x_cpf, x_npf, x_vtx = gaussian_noise(sample=(glob,cpf,npf,vtx), epsilon=5, restrict_impact=-1, epsilon_factors=epsilon_factors)
model.eval()
pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,attack="gaussian-1")
'''