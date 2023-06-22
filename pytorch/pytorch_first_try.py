from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from argparse import ArgumentParser
import shutil
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
from pdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm
import copy

import imp

from attacks import *
from definitions import epsilons_per_feature, vars_per_candidate

glob_vars = vars_per_candidate['glob']
model_output_directory = "/net/scratch_cms3a/hschoenen/deepjet/results/"


def train_loop(dataloader, nbatches, model, loss_fn, optimizer, device, epoch, epoch_pbar, attack, att_magnitude, restrict_impact, epsilon_factors, acc_loss, batchsize=4000, save_batch_progress=[], valgen=None, scheduler=None):
    batch_train_losses = []
    batch_val_losses = []
    # loop over the batches
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
        
        # save initialized model including loss values
        if epoch==0 and len(save_batch_progress)>0:
            if b==0:
                model.eval()
                # compute training loss
                pred = model(glob,cpf,npf,vtx)
                train_loss = loss_fn(pred, y.type_as(pred)).item()
                batch_train_losses.append(train_loss)
                # compute validation loss
                valgen.prepareNextEpoch()
                nbatches_val = valgen.getNBatches()
                val_generator = valgen.feedNumpyData()
                val_loss = val_loop(val_generator, nbatches_val, model, loss_fn, device, epoch, batchsize=batchsize)
                batch_val_losses.append(val_loss)
                # save model parameters
                checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict(),'epoch': epoch,'scheduler': scheduler.state_dict(),'best_loss': None,'train_loss': train_loss,'val_loss': val_loss}
                torch.save(checkpoint, model_output_directory+'/checkpoint_epoch_1_batch_0.pth')
                model.train()
                
        # Compute prediction and loss
        pred = model(glob,cpf,npf,vtx)
        loss = loss_fn(pred, y.type_as(pred))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Add loss to accumulated loss and calculate average batch loss
        acc_loss += loss.item()
        avg_loss = acc_loss / (b + len(pred[:,0])/batchsize)
        # Update progress bar description
        desc = f'Epoch {epoch+1} - loss {avg_loss:.6f}'
        epoch_pbar.set_description(desc)
        epoch_pbar.update(1)
        
        # optional: save progress at the batches specified in save_batch_progress=[]
        if epoch==0:
            batch_train_losses.append(loss.item())
            if b+1 in save_batch_progress:
                model.eval()
                # compute validation loss
                valgen.prepareNextEpoch()
                nbatches_val = valgen.getNBatches()
                val_generator = valgen.feedNumpyData()
                val_loss = val_loop(val_generator, nbatches_val, model, loss_fn, device, epoch, batchsize=batchsize)
                batch_val_losses.append(val_loss)
                # save model parameters
                checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict(),'epoch': epoch,'scheduler': scheduler.state_dict(),'best_loss': None,'train_loss': avg_loss,'val_loss': val_loss}
                torch.save(checkpoint, model_output_directory+'/checkpoint_epoch_'+str(epoch+1)+'_batch_'+str(b+1)+'.pth')
                model.train()
            else:
                batch_val_losses.append(0)
    if epoch==0:
        np.save('{}/loss_values/batch_training_losses.npy'.format(model_output_directory),np.array(batch_train_losses))
        np.save('{}/loss_values/batch_validation_losses.npy'.format(model_output_directory),np.array(batch_val_losses))
    return avg_loss


def val_loop(dataloader, nbatches, model, loss_fn, device, epoch, batchsize=4000):
    num_batches = nbatches
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        # loop over batches
        for b in range(nbatches):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            features_list, truth_list = next(dataloader)
            glob = torch.Tensor(features_list[0]).to(device)
            cpf = torch.Tensor(features_list[1]).to(device)
            npf = torch.Tensor(features_list[2]).to(device)
            vtx = torch.Tensor(features_list[3]).to(device)
            y = torch.Tensor(truth_list[0]).to(device)
            # set global defaults to zero
            glob[:,:] = torch.where(glob[:,:] == -999., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])
            glob[:,:] = torch.where(glob[:,:] ==   -1., torch.zeros(len(glob),glob_vars).to(device), glob[:,:])
            # compute prediction and loss
            pred = model(glob, cpf, npf, vtx)
            test_loss += loss_fn(pred, y.type_as(pred)).item()
            avg_loss = test_loss / (b + len(pred[:,0])/batchsize)
            # compute number of correct predictions
            _, labels = y.max(dim=1)
            total += cpf.shape[0]
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    # compute accuracy
    correct /= total # this is dividing by the total length of validation set, also works for the edge case of the last batch
    print(f"Test Error: \n Accuracy: {(100*correct):>0.6f}%, Avg loss: {avg_loss:>6f} \n")
    return avg_loss

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

class training_base(object):
    
    def __init__(self, model = None, criterion = cross_entropy_one_hot, optimizer = None, scheduler = None, splittrainandtest=0.85, useweights=False, parser=None, resume_silently=False, recreate_silently=False, evaluation_inputFile=''):

        # build training base for training
        if evaluation_inputFile=='':
            import sys
            scriptname = sys.argv[0]
            parser = ArgumentParser('Run the training')
            parser.add_argument('inputDataCollection')
            parser.add_argument('outputDir')
            parser.add_argument("--submitbatch",  help="submits the job to condor" , default=False, action="store_true")
            parser.add_argument("--walltime",  help="sets the wall time for the batch job, format: 1d5h or 2d or 3h etc" , default='1d')
            parser.add_argument("--isbatchrun",   help="is batch run", default=False, action="store_true")
            args = parser.parse_args()
            self.inputData = os.path.abspath(args.inputDataCollection)
            self.outputDir = args.outputDir
        # just build training base for model evaluation
        else:
            self.inputData = os.path.abspath(evaluation_inputFile)
            self.outputDir = ''
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainedepoches = 0
        self.best_loss = np.inf
        self.checkpoint = 0

        self.epoch_train_losses = []
        self.epoch_val_losses = []
        
        # make a global variable for the model output directory
        global model_output_directory
        model_output_directory = self.outputDir
    
        isNewTraining=True
        if self.outputDir!='':
            if os.path.isdir(self.outputDir):
                # ask user, if old training should be continued
                if not (resume_silently or recreate_silently):
                    var = input('output dir exists. To recover a training, please type "yes"\n')
                    if not var == 'yes':
                        raise Exception('output directory must not exist yet')
                    isNewTraining=False
                # continue old training automatically (no terminal interaction required)
                if resume_silently:
                    isNewTraining=False     
            else:
                # create output directory
                if (resume_silently or recreate_silently):
                    raise Exception('there is no old training, which can be continued or overwritten')
                os.mkdir(self.outputDir)
            self.outputDir = os.path.abspath(self.outputDir)
            self.outputDir+='/'

            # delete progress and start new training automatically (no terminal interaction required)
            if recreate_silently:
                isNewTraining=True
                os.system('rm -rf '+ self.outputDir +'*')
            
            # copy configuration to output dir
            if not args.isbatchrun:
                try:
                    shutil.copyfile(scriptname,self.outputDir+os.path.basename(scriptname))
                except shutil.SameFileError:
                    pass
                except BaseException as e:
                    raise e
                self.copied_script = self.outputDir+os.path.basename(scriptname)
            else:
                self.copied_script = scriptname

        # create training and validation datasets
        self.train_data = DataCollection()
        self.train_data.readFromFile(self.inputData)
        self.train_data.useweights = useweights
        if splittrainandtest>0:
            self.val_data = self.train_data.split(splittrainandtest)
        
        # load model checkpoint, if training should be continued
        if not isNewTraining:
            if os.path.isfile(self.outputDir+'/checkpoint.pth'):
                kfile = self.outputDir+'/checkpoint.pth' 
            if os.path.isfile(kfile):
                print(kfile)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.checkpoint = torch.load(kfile)
                self.trainedepoches = self.checkpoint['epoch']
                self.best_loss = self.checkpoint['best_loss']
                self.model.load_state_dict(self.checkpoint['state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                self.scheduler.load_state_dict(self.checkpoint['scheduler'])
                self.epoch_train_losses = np.load('{}/loss_values/epoch_training_losses.npy'.format(model_output_directory)).tolist()
                self.epoch_val_losses = np.load('{}/loss_values/epoch_validation_losses.npy'.format(model_output_directory)).tolist()
            else:
                print('no model found in existing output dir, starting training from scratch')
            
    def saveModel(self,model, optimizer, epoch, scheduler, best_loss, train_loss, val_loss, is_best = False):
        checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict(),'epoch': epoch,'scheduler': scheduler.state_dict(),'best_loss': best_loss,'train_loss': train_loss,'val_loss': val_loss}
        if is_best:
            torch.save(checkpoint, self.outputDir+'checkpoint_best_loss.pth')
        else:
            torch.save(checkpoint, self.outputDir+'checkpoint.pth')
        torch.save(checkpoint, self.outputDir+'checkpoint_epoch_'+str(epoch)+'.pth')
        
    def _initTraining(self, batchsize, use_sum_of_squares=False):
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        self.train_data.batch_uses_sum_of_squares=use_sum_of_squares
        self.val_data.batch_uses_sum_of_squares=use_sum_of_squares
        self.train_data.writeToFile(self.outputDir+'trainsamples.djcdc')
        self.val_data.writeToFile(self.outputDir+'valsamples.djcdc')   
        self.train_data.setBatchSize(batchsize)
        self.val_data.setBatchSize(batchsize)
        
    def trainModel(self, nepochs, batchsize, batchsize_use_sum_of_squares = False, extend_truth_list_by=0,
                   load_in_mem = False, max_files = -1, plot_batch_loss = False, attack = None, att_magnitude = 0., restrict_impact = -1, **trainargs):
        self._initTraining(batchsize, batchsize_use_sum_of_squares)
        print('starting training')
        if load_in_mem:
            print('make features')
            X_train = self.train_data.getAllFeatures(nfiles=max_files)
            X_test = self.val_data.getAllFeatures(nfiles=max_files)
            print('make truth')
            Y_train = self.train_data.getAllLabels(nfiles=max_files)
            Y_test = self.val_data.getAllLabels(nfiles=max_files)
            self.keras_model.fit(X_train, Y_train, batch_size=batchsize, epochs=nepochs,
                                 callbacks=self.callbacks.callbacks,
                                 validation_data=(X_test, Y_test),
                                 max_queue_size=1,
                                 use_multiprocessing=False,
                                 workers=0,    
                                 **trainargs)
        else:
            #prepare generator 
            print("setting up generator... can take a while")
            traingen = self.train_data.invokeGenerator()
            valgen = self.val_data.invokeGenerator()
            traingen.setBatchSize(batchsize)
            valgen.setBatchSize(batchsize)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            #load epsilon factors (specified paths in definitions.py)
            epsilon_factors = {
                'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(self.device),
                'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(self.device),
                'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(self.device),
                'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(self.device),
            }
            # create loss_values directory
            if not os.path.isdir(model_output_directory+'/loss_values'):
                os.system('mkdir '+model_output_directory+'/loss_values')
                
            # loop over epochs
            while(self.trainedepoches < nepochs):
                traingen.prepareNextEpoch()
                valgen.prepareNextEpoch()
                # get number of training and validation batches
                nbatches_train = traingen.getNBatches() #might have changed due to shuffeling
                nbatches_val = valgen.getNBatches()
                
                train_generator=traingen.feedNumpyData()
                val_generator=valgen.feedNumpyData()
            
                print('>>>> epoch', self.trainedepoches,"/",nepochs)
                print('training batches: ',nbatches_train)
                print('validation batches: ',nbatches_val)
                
                with tqdm(total = nbatches_train) as epoch_pbar:
                    epoch_pbar.set_description(f'Epoch {self.trainedepoches + 1}')
                    self.model.train()
                    for param_group in self.optimizer.param_groups:
                        print('/n Learning rate = '+str(param_group['lr'])+' /n')
                        
                    # compute training loss and perform gradient descent
                    train_loss = train_loop(train_generator, nbatches_train, self.model, self.criterion, self.optimizer, self.device, self.trainedepoches, epoch_pbar, attack, att_magnitude, restrict_impact, epsilon_factors, acc_loss=0, batchsize=batchsize, save_batch_progress=[1,2,3,4,5,10,20,50,100,200,500,1000,2000,5000], valgen=valgen, scheduler=self.scheduler)
                    self.epoch_train_losses.append(train_loss)
                    np.save('{}/loss_values/epoch_training_losses.npy'.format(model_output_directory),np.array(self.epoch_train_losses))
                    self.scheduler.step()
                    
                    # compute validation loss and accuracy
                    self.model.eval()
                    valgen.prepareNextEpoch()
                    nbatches_val = valgen.getNBatches()
                    val_generator = valgen.feedNumpyData()
                    val_loss = val_loop(val_generator, nbatches_val, self.model, self.criterion, self.device, self.trainedepoches, batchsize=batchsize)
                    self.epoch_val_losses.append(val_loss)
                    np.save('{}/loss_values/epoch_validation_losses.npy'.format(model_output_directory),np.array(self.epoch_val_losses))
                    
                    self.trainedepoches += 1
                    
                    # save model with the best validation loss as checkpoint_best_loss.pth
                    if(val_loss < self.best_loss):
                        self.best_loss = val_loss
                        self.saveModel(self.model, self.optimizer, self.trainedepoches, self.scheduler, self.best_loss, train_loss, val_loss, is_best = True)
                        best_epoch = self.trainedepoches
                    # save model as checkpoint_epoch_{}.pth
                    self.saveModel(self.model, self.optimizer, self.trainedepoches, self.scheduler, self.best_loss, train_loss, val_loss, is_best = False)
                traingen.shuffleFileList()
                
            # save best epoch
            self.epoch_train_losses.append(best_epoch)
            self.epoch_val_losses.append(best_epoch)
            np.save('{}/loss_values/epoch_training_losses.npy'.format(model_output_directory),np.array(self.epoch_train_losses))
            np.save('{}/loss_values/epoch_validation_losses.npy'.format(model_output_directory),np.array(self.epoch_val_losses))
