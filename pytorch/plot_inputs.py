# Hendrik Schönen
# This script is supposed to get the training data and save the input features before and after applying an attack
# It has to be executed like: python3 pytorch/plot_inputs.py /eos/user/h/heschone/DeepJet/DeepJet_Run2/Data/dataCollection.djcdc /eos/user/h/heschone/DeepJet/Train_DF_Run2/trash/test141

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# train_deepflavour imports
import torch 
import torch.nn as nn
from pytorch_first_try import training_base
from pytorch_deepjet import *
from pytorch_deepjet_run2 import *
from pytorch_deepjet_transformer import DeepJetTransformer
from pytorch_ranger import Ranger
# pytorch_first_try imports
import sys
import os
from argparse import ArgumentParser
import shutil
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
from pdb import set_trace
import numpy as np
import torch.nn.functional as F
#from torch.optim import Adam, SGD
from tqdm import tqdm
import copy
import imp
from attacks import *
from definitions import epsilons_per_feature, vars_per_candidate
# attacks imports
from definitions import *

print("imports finished")

model_name='nominal'#'fgsm'

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

num_epochs = 60
lr_epochs = max(1, int(num_epochs * 0.3))
lr_rate = 0.01 ** (1.0 / lr_epochs)
mil = list(range(num_epochs - lr_epochs, num_epochs))
model = DeepJet_Run2(num_classes = 6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = cross_entropy_one_hot
optimizer = Ranger(model.parameters(), lr = 5e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = mil, gamma = lr_rate)

train=training_base(model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, testrun=False)
train.train_data.maxFilesOpen=1
print("training_base created")

# Processing in training_base.TrainModel()
batchsize=40000
train.train_data.setBatchSize(batchsize)
train.train_data.batch_uses_sum_of_squares=False
train.train_data.setBatchSize(batchsize)
traingen = train.train_data.invokeGenerator()
traingen.setBatchSize(batchsize)
traingen.prepareNextEpoch()
train_generator=traingen.feedNumpyData()
print("train_generator created")
epsilon_factors = {
                'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(device),
                'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(device),
                'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(device),
                'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(device),
            }
restrict_impact = 0.2

# Load model
check = torch.load('/eos/user/h/heschone/DeepJet/Train_DF_Run2/{}/checkpoint_best_loss.pth'.format(model_name), map_location=torch.device('cpu'))
model.load_state_dict(check['state_dict'])
model.to(device)
model.eval()

def save_features(inputs, targets, outputs, filename="features"):
    print(filename)
    x_glob, x_cpf, x_npf, x_vtx = inputs[0], inputs[1], inputs[2], inputs[3]
    jet_pt = x_glob[:,0].clone().detach().numpy()
    jet_eta = x_glob[:,1].clone().detach().numpy()
    nCpfcand = x_glob[:,2].clone().detach().numpy()
    nNpfcand = x_glob[:,3].clone().detach().numpy()
    sv_pt = x_vtx[:,0,0].clone().detach().numpy()
    sv_deltaR = x_vtx[:,0,1].clone().detach().numpy()
    sv_mass = x_vtx[:,0,2].clone().detach().numpy()
    sv_dxy = x_vtx[:,0,6].clone().detach().numpy()
    sv_dxysig = x_vtx[:,0,7].clone().detach().numpy()
    sv_d3d = x_vtx[:,0,8].clone().detach().numpy()
    sv_d3dsig = x_vtx[:,0,9].clone().detach().numpy()
    isB,isBB,isLeptonicB,isC,isUDS,isG = targets[:,0].clone().detach().numpy(), targets[:,1].clone().detach().numpy(), targets[:,2].clone().detach().numpy(), targets[:,3].clone().detach().numpy(), targets[:,4].clone().detach().numpy(), targets[:,5].clone().detach().numpy()
    prob_isB,prob_isBB,prob_isLeptonicB,prob_isC,prob_isUDS,prob_isG = outputs[:,0].clone().detach().numpy(), outputs[:,1].clone().detach().numpy(), outputs[:,2].clone().detach().numpy(), outputs[:,3].clone().detach().numpy() ,outputs[:,4].clone().detach().numpy(),outputs[:,5].clone().detach().numpy()
    variables = np.array([jet_pt,jet_eta,nCpfcand,nNpfcand,sv_pt,sv_deltaR,sv_mass,sv_dxy,sv_dxysig,sv_d3d,sv_d3dsig, isB,isBB,isLeptonicB,isC,isUDS,isG,prob_isB,prob_isBB,prob_isLeptonicB,prob_isC,prob_isUDS,prob_isG])
    names = ['jet_pt','jet_eta','nCpfcand','nNpfcand','sv_pt','sv_deltaR','sv_mass','sv_dxy','sv_dxysig','sv_d3d','sv_d3dsig', 'isB','isBB','isLeptonicB','isC','isUDS','isG','prob_isB','prob_isBB','prob_isLeptonicB','prob_isC','prob_isUDS','prob_isG']
    os.system('mkdir -p '+"/eos/user/h/heschone/DeepJet/Train_DF_Run2/{}/variables".format(model_name))
    np.save("/eos/user/h/heschone/DeepJet/Train_DF_Run2/{}/variables/variables{}.npy".format(model_name,filename),np.array(variables))
    np.save("/eos/user/h/heschone/DeepJet/Train_DF_Run2/{}/variables/names{}.npy".format(model_name,filename),names)
    
def prediction_surface(inputs, index_jet=0, N_steps=10, dx=0.01, dy=0.01, filename='jet'):
    print(filename)
    x_glob, x_cpf, x_npf, x_vtx = inputs[0], inputs[1], inputs[2], inputs[3]
    one_glob = x_glob[index_jet:index_jet+2,:].clone().detach()
    one_cpf = x_cpf[index_jet:index_jet+2,:,:].clone().detach()
    one_npf = x_npf[index_jet:index_jet+2,:,:].clone().detach()
    one_vtx = x_vtx[index_jet:index_jet+2,:,:].clone().detach()
    x_initial = one_glob[0,0].item()
    y_initial = one_glob[0,1].item()

    N_nodes=(2*N_steps+1)**2
    prediction_array = torch.zeros(N_nodes,6)
    x_array = torch.zeros(N_nodes)
    y_array = torch.zeros(N_nodes)
    
    n_step=0
    for i in range(-N_steps,N_steps+1):
        x = x_initial + i*dx
        for j in range(-N_steps,N_steps+1):
            if n_step%100==0:
                print(n_step)
                print(x_initial," ",dx," , ",y_initial," ",dy)
            y = y_initial + j*dy
            one_glob[0,0] = x
            one_glob[0,1] = y
            pred = model(one_glob, one_cpf, one_npf, one_vtx)
            prediction_array[n_step,:] = pred[0,:]
            x_array[n_step] = x
            y_array[n_step] = y
            n_step += 1
    np.save("/eos/user/h/heschone/DeepJet/Train_DF_Run2/{}/variables/{}_grid_loss.npy".format(model_name, filename),prediction_array.detach().numpy())
    np.save("/eos/user/h/heschone/DeepJet/Train_DF_Run2/{}/variables/{}_grid_pt.npy".format(model_name, filename),x_array.detach().numpy())
    np.save("/eos/user/h/heschone/DeepJet/Train_DF_Run2/{}/variables/{}_grid_eta.npy".format(model_name, filename),y_array.detach().numpy())
    
# Processing the data in train_loop()
for b in range(1):
    if b%100==0:
        print("b=",b)
    features_list, truth_list = next(train_generator)
    glob = torch.Tensor(features_list[0]).to(device)
    cpf = torch.Tensor(features_list[1]).to(device)
    npf = torch.Tensor(features_list[2]).to(device)
    vtx = torch.Tensor(features_list[3]).to(device)
    y = torch.Tensor(truth_list[0]).to(device)
    pred = (model(glob,cpf,npf,vtx)).detach()
    save_features((glob,cpf,npf,vtx),y,pred,filename="")
    
    # save loss surface for a common bjet
    for index_jet in range(len(glob[:,0])):
        if (y[index_jet,0]==1) and (pred[index_jet,0]>0.5):
            print(index_jet)
            prediction_surface((glob,cpf,npf,vtx), index_jet=index_jet, N_steps=50, dx=0.01, dy=0.001, filename='bjet')
            break
            
    # FGSM epsilon=0.01
    x_glob, x_cpf, x_npf, x_vtx = fgsm_attack(sample=(glob,cpf,npf,vtx), 
                                              epsilon=0.01,
                                              dev=device,
                                              targets=y,
                                              thismodel=model,
                                              thiscriterion=train.criterion,
                                              restrict_impact=restrict_impact,
                                              epsilon_factors=epsilon_factors,
                                              allow_zeros=True)
    pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
    save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,filename="_fgsm")
    
    # save loss surface for a zero gradient jet
    for index_jet in range(len(glob[:,0])):
        if (x_glob[index_jet,0]==glob[index_jet,0]):
            print(index_jet)
            prediction_surface((glob,cpf,npf,vtx), index_jet=index_jet, N_steps=50, dx=0.01, dy=0.001, filename='zerogradjet')
            break
    
    # FGSM flavour epsilons=(0.01,0.01,0.01)
    x_glob, x_cpf, x_npf, x_vtx = fgsm_attack_flavour(sample=(glob,cpf,npf,vtx), 
                                                      epsilons=[0.01,0.01,0.01],
                                                      dev=device,
                                                      targets=y,
                                                      thismodel=model,
                                                      thiscriterion=train.criterion,
                                                      restrict_impact=restrict_impact,
                                                      epsilon_factors=epsilon_factors)
    pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
    save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,filename="_fgsm_flavour_default")
    
    # FGSM flavour epsilons=(0.012,0.008,0.01)
    x_glob, x_cpf, x_npf, x_vtx = fgsm_attack_flavour(sample=(glob,cpf,npf,vtx), 
                                                      epsilons=[0.012,0.008,0.01],
                                                      dev=device,
                                                      targets=y,
                                                      thismodel=model,
                                                      thiscriterion=train.criterion,
                                                      restrict_impact=restrict_impact,
                                                      epsilon_factors=epsilon_factors)
    pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
    save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,filename="_fgsm_flavour_012008010")
    
    # FGSM domain epsilons=default
    x_glob, x_cpf, x_npf, x_vtx = fgsm_attack_domain(sample=(glob,cpf,npf,vtx), 
                                                     epsilon="default",
                                                     dev=device,
                                                     targets=y,
                                                     thismodel=model,
                                                     thiscriterion=train.criterion,
                                                     restrict_impact=restrict_impact,
                                                     epsilon_factors=epsilon_factors,
                                                     allow_zeros=True)
    pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
    save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,filename="_fgsm_domain_default")
    
    # FGSM domain epsilons=random
    x_glob, x_cpf, x_npf, x_vtx = fgsm_attack_domain(sample=(glob,cpf,npf,vtx), 
                                                     epsilon="random",
                                                     dev=device,
                                                     targets=y,
                                                     thismodel=model,
                                                     thiscriterion=train.criterion,
                                                     restrict_impact=restrict_impact,
                                                     epsilon_factors=epsilon_factors,
                                                     allow_zeros=True)
    pred = (model(x_glob,x_cpf,x_npf,x_vtx)).detach()
    save_features((x_glob,x_cpf,x_npf,x_vtx),y,pred,filename="_fgsm_domain_random")
    