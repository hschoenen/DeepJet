#!/usr/bin/env python3

# this script computes model predictions and saves them

# python3 pytorch/predict_pytorch.py DeepJet_Run2 /net/scratch_cms3a/hschoenen/deepjet/results/nominal /checkpoint_best_loss.pth /net/scratch_cms3a/hschoenen/deepjet/results/nominal/trainsamples.djcdc one_sample.txt /net/scratch_cms3a/hschoenen/deepjet/results/nominal/predict_fgsm -attack FGSM -att_magnitude 0.01 -restrict_impact 0.2

# parse arguments
from argparse import ArgumentParser
parser = ArgumentParser('Apply a model to a (test) source sample.')
# model architecture: DeepJet_Run2
parser.add_argument('model')
# model parameters: /net/scratch_cms3a/hschoenen/deepjet/results/nominal /checkpoint_best_loss.pth
parser.add_argument('inputModel')
# training data collection: /net/scratch_cms3a/hschoenen/deepjet/results/nominal/trainsamples.djcdc
parser.add_argument('trainingDataCollection', help="the training data collection. Used to infer data format and batch size.")
# test data filelist: one_sample.txt 
parser.add_argument('inputSourceFileList', help="can be text file or a DataCollection file in the same directory as the sample files, or just a single traindata file.")
# output directory: /net/scratch_cms3a/hschoenen/deepjet/results/nominal/predict_nominal
parser.add_argument('outputDir', help="will be created if it doesn't exist.")
parser.add_argument("-b", help="batch size, overrides the batch size from the training data collection.",default="-1")
parser.add_argument("--gpu",  help="select specific GPU", metavar="OPT", default="")
parser.add_argument("--unbuffered", help="do not read input in memory buffered mode (for lower memory consumption on fast disks)", default=False, action="store_true")
parser.add_argument("--pad_rowsplits", help="pad the row splits if the input is ragged", default=False, action="store_true")
# attack type: FGSM, FGSM_flavour, FGSM_domain
parser.add_argument("-attack", help="use adversarial attack (Noise|FGSM) or leave blank to use undisturbed features only", default="")
# FGSM epsilon
parser.add_argument("-att_magnitude", help="distort input features with adversarial attack, using specified magnitude of attack", default="-1")
# restrict impact: 0.2
parser.add_argument("-restrict_impact", help="limit attack impact to this fraction of the input value (percent-cap on distortion)", default="-1")
# FGSM epsilon by flavour
parser.add_argument("-att_magnitude_flavour", help="distort input features with adversarial attack, using specified magnitude of attack for each flavour", nargs=3, type=float, default=[0,0,0])
# FGSM epsilon by (flavour,pT,eta) domain
parser.add_argument("-att_magnitude_domain", help="distort input features with adversarial attack, using specified magnitude of attack for each (flavour,pT,eta) domain", default="epsilon_tensor")
args = parser.parse_args()
batchsize = int(args.b)
attack = args.attack
if attack=="FGSM":
    att_magnitude = float(args.att_magnitude)
elif attack=="FGSM_flavour":
    att_magnitude = args.att_magnitude_flavour
elif attack=="FGSM_domain":
    att_magnitude = args.att_magnitude_domain
elif attack=="Gaussian":
    att_magnitude = float(args.att_magnitude)
else:
    att_magnitude=0
restrict_impact = float(args.restrict_impact)

import imp
import numpy as np
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
import tempfile
import atexit
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_deepjet import DeepJet
from pytorch_deepjet_run2 import DeepJet_Run2
from pytorch_deepjet_transformer import DeepJetTransformer
from torch.optim import Adam, SGD
from tqdm import tqdm
from attacks import apply_noise, fgsm_attack, fgsm_attack_flavour, fgsm_attack_domain, gaussian_noise
from definitions import epsilons_per_feature, vars_per_candidate

glob_vars = vars_per_candidate['glob']

inputdatafiles=[]
inputdir=None

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

def test_loop(dataloader, model, nbatches, pbar, attack = "", att_magnitude = -1., restrict_impact = -1., loss_fn = cross_entropy_one_hot, epsilon_factors=None):
    predictions = 0
    for b in range(nbatches):
        features_list, truth_list = next(dataloader)
        glob = torch.Tensor(features_list[0]).to(device)
        cpf = torch.Tensor(features_list[1]).to(device)
        npf = torch.Tensor(features_list[2]).to(device)
        vtx = torch.Tensor(features_list[3]).to(device)
        y = torch.Tensor(truth_list[0]).to(device)
        # set global default values to zero
        glob[:,:] = torch.where(glob[:,:] == -999., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])
        glob[:,:] = torch.where(glob[:,:] ==   -1., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])
        
        # apply attack
        if attack == 'Noise':
            glob = apply_noise(glob, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='glob')
            cpf = apply_noise(cpf, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='cpf')
            npf = apply_noise(npf, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='npf')
            vtx = apply_noise(vtx, 
                               magn=att_magnitude,
                               offset=[0],
                               dev=device,
                               restrict_impact=restrict_impact,
                               var_group='vtx')
        elif attack == 'FGSM':
            glob, cpf, npf, vtx = fgsm_attack(sample=(glob,cpf,npf,vtx), 
                                               epsilon=att_magnitude,
                                               dev=device,
                                               targets=y,
                                               thismodel=model,
                                               thiscriterion=loss_fn,
                                               restrict_impact=restrict_impact,
                                               epsilon_factors=epsilon_factors)
        
        elif attack == 'FGSM_flavour':
            glob, cpf, npf, vtx = fgsm_attack_flavour(sample=(glob,cpf,npf,vtx), 
                                                      epsilons=att_magnitude,
                                                      dev=device,
                                                      targets=y,
                                                      thismodel=model,
                                                      thiscriterion=loss_fn,
                                                      restrict_impact=restrict_impact,
                                                      epsilon_factors=epsilon_factors)
            
        elif attack == 'FGSM_domain':
            glob, cpf, npf, vtx = fgsm_attack_domain(sample=(glob,cpf,npf,vtx), 
                                                      epsilon=att_magnitude,
                                                      dev=device,
                                                      targets=y,
                                                      thismodel=model,
                                                      thiscriterion=loss_fn,
                                                      restrict_impact=restrict_impact,
                                                      epsilon_factors=epsilon_factors)

        elif attack == 'Gaussian':
            glob, cpf, npf, vtx = gaussian_noise(sample=(glob,cpf,npf,vtx), 
                                                 epsilon=att_magnitude,
                                                 restrict_impact=restrict_impact,
                                                 epsilon_factors=epsilon_factors)
            
        # Compute prediction
        pred = nn.Softmax(dim=1)(model(glob,cpf,npf,vtx)).cpu().detach().numpy()
        if b == 0:
            predictions = pred
        else:
            predictions = np.concatenate((predictions, pred), axis=0)
        desc = 'Predicting probs : '
        pbar.set_description(desc)
        pbar.update(1)
    return predictions

# prepare input lists for different file formats
if args.inputSourceFileList[-6:] == ".djcdc":
    print('reading from data collection',args.inputSourceFileList)
    predsamples = DataCollection(args.inputSourceFileList)
    inputdir = predsamples.dataDir
    for s in predsamples.samples:
        inputdatafiles.append(s)  
elif args.inputSourceFileList[-6:] == ".djctd":
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    infile = os.path.basename(args.inputSourceFileList)
    inputdatafiles.append(infile)
else:
    print('reading from text file',args.inputSourceFileList)
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    with open(args.inputSourceFileList, "r") as f:
        for s in f:
            inputdatafiles.append(s.replace('\n', '').replace(" ",""))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.enabled=False

# select model
if args.model == 'DeepJet':
    model = DeepJet(num_classes = 6)
if args.model == 'DeepJet_Run2':
    model = DeepJet_Run2(num_classes = 6)
if args.model == 'DeepJetTransformer':
    model = DeepJetTransformer(num_classes = 4)
# load model
check = torch.load(args.inputModel, map_location=torch.device('cpu'))
model.load_state_dict(check['state_dict'])
model.to(device)
model.eval()

dc = None
if args.inputSourceFileList[-6:] == ".djcdc" and not args.trainingDataCollection[-6:] == ".djcdc":
    dc = DataCollection(args.inputSourceFileList)
    if batchsize < 1:
        batchsize = 1
    print('No training data collection given. Using batch size of',batchsize)
else:
    dc = DataCollection(args.trainingDataCollection)

outputs = []
os.system('mkdir -p '+args.outputDir)

for inputfile in inputdatafiles:
    use_inputdir = inputdir
    if inputfile[0] == "/":
        use_inputdir = ""
    else:
        use_inputdir = use_inputdir+"/"
    outfilename = "pred_" + os.path.basename(inputfile)
    
    td = dc.dataclass()

    if inputfile[-5:] == 'djctd':
        if args.unbuffered:
            td.readFromFile(use_inputdir+inputfile)
        else:
            td.readFromFileBuffered(use_inputdir+inputfile)
    else:
        print('converting '+inputfile)
        print(use_inputdir+inputfile)
        td.readFromSourceFile(use_inputdir+inputfile, dc.weighterobjects, istraining=False)

    gen = TrainDataGenerator()
    if batchsize < 1:
        batchsize = dc.getBatchSize()
    gen.setBatchSize(batchsize)
    gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)

    with tqdm(total = gen.getNBatches()) as pbar:
        pbar.set_description('Predicting : ')
        
    epsilon_factors = {
                'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(device),
                'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(device),
                'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(device),
                'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(device),
            }
    # compute model predictions
    predicted = test_loop(gen.feedNumpyData(), model, nbatches=gen.getNBatches(), pbar = pbar, attack = attack, att_magnitude = att_magnitude, restrict_impact = restrict_impact, epsilon_factors=epsilon_factors)
    
    x = td.transferFeatureListToNumpy(args.pad_rowsplits)
    w = td.transferWeightListToNumpy(args.pad_rowsplits)
    y = td.transferTruthListToNumpy(args.pad_rowsplits)
    td.clear()
    gen.clear()

    if not type(predicted) == list: #circumvent that keras return only an array if there is just one list item
        predicted = [predicted]   
        
    # Optimal would be to include the discriminators here
    overwrite_outname = td.writeOutPrediction(predicted, x, y, w, args.outputDir + "/" + outfilename, use_inputdir+"/"+inputfile)
    if overwrite_outname is not None:
        outfilename = overwrite_outname
    outputs.append(outfilename)
    
with open(args.outputDir + "/outfiles.txt","w") as f:
    for l in outputs:
        f.write(l+'\n')
