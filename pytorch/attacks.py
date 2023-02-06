from definitions import *
import torch
import numpy as np

def apply_noise(sample, magn=1e-2,offset=[0], dev=torch.device("cpu"), restrict_impact=-1, var_group='glob'):
    if magn == 0:
        return sample

    seed = 0
    np.random.seed(seed)
    with torch.no_grad():
        if var_group == 'glob':
            noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),vars_per_candidate[var_group]))).to(dev)
        else:
            noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),cands_per_variable[var_group],vars_per_candidate[var_group]))).to(dev)
        xadv = sample + noise

        if var_group == 'glob':
            for i in range(vars_per_candidate['glob']):
                if i in integer_per_variable[var_group]:
                    xadv[:,i] = sample[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults = sample[:,i].cpu() == defaults_per_variable[var_group][i]
                    if torch.sum(defaults) != 0:
                        xadv[:,i][defaults] = sample[:,i][defaults]

                    if restrict_impact > 0:
                        difference = xadv[:,i] - sample[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(sample[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv[high_impact,i] = sample[high_impact,i] + allowed_perturbation[high_impact] * torch.sign(noise[high_impact,i])

        else:
            for j in range(cands_per_variable[var_group]):
                for i in range(vars_per_candidate[var_group]):
                    if i in integer_variables_by_candidate[var_group]:
                        xadv[:,j,i] = sample[:,j,i]
                    else:
                        defaults = sample[:,j,i].cpu() == defaults_per_variable[var_group][i]
                        if torch.sum(defaults) != 0:
                            xadv[:,j,i][defaults] = sample[:,j,i][defaults]

                        if restrict_impact > 0:
                            difference = xadv[:,j,i] - sample[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(sample[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv[high_impact,j,i] = sample[high_impact,j,i] + allowed_perturbation[high_impact] * torch.sign(noise[high_impact,j,i])       

        return xadv

def fgsm_attack(epsilon=1e-2,sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, dev=torch.device("cpu"), restrict_impact=-1, epsilon_factors=None, allow_zeros=True):
    if epsilon == 0:
        return sample

    glob, cpf, npf, vtx = sample
    xadv_glob = glob.clone().detach()
    xadv_cpf = cpf.clone().detach()
    xadv_npf = npf.clone().detach()
    xadv_vtx = vtx.clone().detach()

    xadv_glob.requires_grad = True
    xadv_cpf.requires_grad = True
    xadv_npf.requires_grad = True
    xadv_vtx.requires_grad = True

    preds = thismodel(xadv_glob,xadv_cpf,xadv_npf,xadv_vtx)

    loss = thiscriterion(preds, targets)

    thismodel.zero_grad()
    loss.backward()

    with torch.no_grad():
        dx_glob = torch.sign(xadv_glob.grad.detach())
        dx_cpf = torch.sign(xadv_cpf.grad.detach())
        dx_npf = torch.sign(xadv_npf.grad.detach())
        dx_vtx = torch.sign(xadv_vtx.grad.detach())

        xadv_glob += epsilon * epsilon_factors['glob'] * dx_glob
        xadv_cpf += epsilon * epsilon_factors['cpf'] * dx_cpf
        xadv_npf += epsilon * epsilon_factors['npf'] * dx_npf
        xadv_vtx += epsilon * epsilon_factors['vtx'] * dx_vtx
        
        '''
        # investigate zero gradient jets and check if it is due to defaultvalues
        if (glob[:,0]==xadv_glob[:,0]).nonzero().size(dim=0)>0:
            zeros = (glob[:,0]==xadv_glob[:,0]).nonzero()
            for i in range(len(zeros)):
                print('default values for index ',zeros[i][0])
                sum_defaults_glob = 0
                for j in range(vars_per_candidate['glob']):
                    defaults_glob = glob[i,j].cpu() == defaults_per_variable['glob'][j]
                    sum_defaults_glob = len(defaults_glob.nonzero())
                print(sum_defaults_glob)
                sum_defaults_cpf = np.zeros(cands_per_variable['cpf'])
                for j in range(cands_per_variable['cpf']):
                    for k in range(vars_per_candidate['cpf']):
                        defaults_cpf = cpf[i,j,k].cpu() == defaults_per_variable['cpf'][k]
                        sum_defaults_cpf[j] += len(defaults_cpf.nonzero())
                print(sum_defaults_cpf)
                sum_defaults_npf = np.zeros(cands_per_variable['npf'])
                for j in range(cands_per_variable['npf']):
                    for k in range(vars_per_candidate['npf']):
                        defaults_npf = npf[i,j,k].cpu() == defaults_per_variable['npf'][k]
                        sum_defaults_npf[j] += len(defaults_npf.nonzero())
                print(sum_defaults_npf)
                sum_defaults_vtx = np.zeros(cands_per_variable['vtx'])
                for j in range(cands_per_variable['vtx']):
                    for k in range(vars_per_candidate['vtx']):
                        defaults_vtx = vtx[i,j,k].cpu() == defaults_per_variable['vtx'][k]
                        sum_defaults_vtx[j] += len(defaults_vtx.nonzero())
                print(sum_defaults_vtx)
        '''
        if (glob[:,0]==xadv_glob[:,0]).nonzero().size(dim=0)>0 and allow_zeros==False:
            zeros = (glob[:,0]==xadv_glob[:,0]).nonzero()
            print("For index ",zeros[0].item())
            print(dx_glob[zeros[0]-1:zeros[0]+2,:])
            raise ValueError("no change after applying fgsm attack")
        
        if reduced:
            for i in range(vars_per_candidate['glob']):
                if i in integer_variables_by_candidate['glob']:
                    xadv_glob[:,i] = glob[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:,i].cpu() == defaults_per_variable['glob'][i]
                    if torch.sum(defaults_glob) != 0:
                        xadv_glob[:,i][defaults_glob] = glob[:,i][defaults_glob]

                    if restrict_impact > 0:
                        difference = xadv_glob[:,i] - glob[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(glob[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv_glob[high_impact,i] = glob[high_impact,i] + allowed_perturbation[high_impact] * dx_glob[high_impact,i]

            for j in range(cands_per_variable['cpf']):
                for i in range(vars_per_candidate['cpf']):
                    if i in integer_variables_by_candidate['cpf']:
                        xadv_cpf[:,j,i] = cpf[:,j,i]
                    else:
                        defaults_cpf = cpf[:,j,i].cpu() == defaults_per_variable['cpf'][i]
                        if torch.sum(defaults_cpf) != 0:
                            xadv_cpf[:,j,i][defaults_cpf] = cpf[:,j,i][defaults_cpf]

                        if restrict_impact > 0:
                            difference = xadv_cpf[:,j,i] - cpf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(cpf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_cpf[high_impact,j,i] = cpf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_cpf[high_impact,j,i]        

            for j in range(cands_per_variable['npf']):
                for i in range(vars_per_candidate['npf']):
                    if i in integer_variables_by_candidate['npf']:
                        xadv_npf[:,j,i] = npf[:,j,i]
                    else:
                        defaults_npf = npf[:,j,i].cpu() == defaults_per_variable['npf'][i]
                        if torch.sum(defaults_npf) != 0:
                            xadv_npf[:,j,i][defaults_npf] = npf[:,j,i][defaults_npf]

                        if restrict_impact > 0:
                            difference = xadv_npf[:,j,i] - npf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(npf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_npf[high_impact,j,i] = npf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_npf[high_impact,j,i]   

            for j in range(cands_per_variable['vtx']):
                for i in range(vars_per_candidate['vtx']):
                    if i in integer_variables_by_candidate['vtx']:
                        xadv_vtx[:,j,i] = vtx[:,j,i]
                    else:
                        defaults_vtx = vtx[:,j,i].cpu() == defaults_per_variable['vtx'][i]
                        if torch.sum(defaults_vtx) != 0:
                            xadv_vtx[:,j,i][defaults_vtx] = vtx[:,j,i][defaults_vtx]

                        if restrict_impact > 0:
                            difference = xadv_vtx[:,j,i] - vtx[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(vtx[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_vtx[high_impact,j,i] = vtx[high_impact,j,i] + allowed_perturbation[high_impact] * dx_vtx[high_impact,j,i]   

        return xadv_glob.detach(),xadv_cpf.detach(),xadv_npf.detach(),xadv_vtx.detach()

def fgsm_attack_flavour(epsilons=[0,0,0],sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, dev=torch.device("cpu"), restrict_impact=-1, epsilon_factors=None):
    if epsilons == [0,0,0]:
        print("No epsilons given")
        return sample
    if targets==None:
        raise ValueError("No truth labels given")
    
    glob, cpf, npf, vtx = sample
    y = targets
    xadv_glob = glob.clone().detach()
    xadv_cpf = cpf.clone().detach()
    xadv_npf = npf.clone().detach()
    xadv_vtx = vtx.clone().detach()

    xadv_glob.requires_grad = True
    xadv_cpf.requires_grad = True
    xadv_npf.requires_grad = True
    xadv_vtx.requires_grad = True

    preds = thismodel(xadv_glob,xadv_cpf,xadv_npf,xadv_vtx)

    loss = thiscriterion(preds, targets)

    thismodel.zero_grad()
    loss.backward()

    with torch.no_grad():
        dx_glob = torch.sign(xadv_glob.grad.detach())
        dx_cpf = torch.sign(xadv_cpf.grad.detach())
        dx_npf = torch.sign(xadv_npf.grad.detach())
        dx_vtx = torch.sign(xadv_vtx.grad.detach())
        
        batchsize = y.size(dim=0)
        epsilon_vector=epsilons[0]*(y[:,0]+y[:,1]+y[:,2]) + epsilons[1]*(y[:,3]) + epsilons[2]*(y[:,4]+y[:,5])
        
        # make sure the true jet flavour is available (search for 0 entries in epsilon_vector)
        #if len((epsilon_vector==0).nonzero())>0:
            #zeros = (epsilon_vector==0).nonzero()
            #print("For index ",zeros[0].item())
            #print("truth label: ",y[zeros[0]])
            #epsilon_vector=epsilon_vector+(epsilon_vector==0)*0.01
            #print(len(zeros)," epsilons changed from 0 to 0.01")
            #raise ValueError("No truth value available")
        
        # broadcast epsilon into the shapes of glob,cpf,npf,vtx and apply FGSM attack
        xadv_glob+=torch.unsqueeze(epsilon_vector,dim=1).expand(glob.size(dim=0),glob.size(dim=1)) * epsilon_factors['glob'] * dx_glob
        xadv_cpf+=torch.unsqueeze(torch.unsqueeze(epsilon_vector, dim=1),dim=1).expand(cpf.size(dim=0), cpf.size(dim=1), cpf.size(dim=2)) * epsilon_factors['cpf'] * dx_cpf
        xadv_npf+=torch.unsqueeze(torch.unsqueeze(epsilon_vector, dim=1),dim=1).expand(npf.size(dim=0), npf.size(dim=1), npf.size(dim=2)) * epsilon_factors['npf'] * dx_npf
        xadv_vtx+=torch.unsqueeze(torch.unsqueeze(epsilon_vector, dim=1),dim=1).expand(vtx.size(dim=0), vtx.size(dim=1), vtx.size(dim=2)) * epsilon_factors['vtx'] * dx_vtx
        
        ''' for loop method (inefficient)
        for i in range(batchsize):
            epsilon=0
            if int(y[i][0].item()+y[i][1].item()+y[i][2].item())==1:
                epsilon=epsilons[0]
            elif int(y[i][3].item())==1:
                epsilon=epsilons[1]
            elif int(y[i][4].item()+y[i][5].item())==1:
                epsilon=epsilons[2]
            else:
                print(" No truth available at index {}".format(i))
                print(y[i])
                raise ValueError('No Truth available')
            xadv_glob[i] += epsilon * epsilon_factors['glob'] * dx_glob[i]
            xadv_cpf[i] += epsilon * epsilon_factors['cpf'] * dx_cpf[i]
            xadv_npf[i] += epsilon * epsilon_factors['npf'] * dx_npf[i]
            xadv_vtx[i] += epsilon * epsilon_factors['vtx'] * dx_vtx[i]
        '''

        if reduced:
            for i in range(vars_per_candidate['glob']):
                if i in integer_variables_by_candidate['glob']:
                    xadv_glob[:,i] = glob[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:,i].cpu() == defaults_per_variable['glob'][i]
                    if torch.sum(defaults_glob) != 0:
                        xadv_glob[:,i][defaults_glob] = glob[:,i][defaults_glob]

                    if restrict_impact > 0:
                        difference = xadv_glob[:,i] - glob[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(glob[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv_glob[high_impact,i] = glob[high_impact,i] + allowed_perturbation[high_impact] * dx_glob[high_impact,i]

            for j in range(cands_per_variable['cpf']):
                for i in range(vars_per_candidate['cpf']):
                    if i in integer_variables_by_candidate['cpf']:
                        xadv_cpf[:,j,i] = cpf[:,j,i]
                    else:
                        defaults_cpf = cpf[:,j,i].cpu() == defaults_per_variable['cpf'][i]
                        if torch.sum(defaults_cpf) != 0:
                            xadv_cpf[:,j,i][defaults_cpf] = cpf[:,j,i][defaults_cpf]

                        if restrict_impact > 0:
                            difference = xadv_cpf[:,j,i] - cpf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(cpf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_cpf[high_impact,j,i] = cpf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_cpf[high_impact,j,i]        

            for j in range(cands_per_variable['npf']):
                for i in range(vars_per_candidate['npf']):
                    if i in integer_variables_by_candidate['npf']:
                        xadv_npf[:,j,i] = npf[:,j,i]
                    else:
                        defaults_npf = npf[:,j,i].cpu() == defaults_per_variable['npf'][i]
                        if torch.sum(defaults_npf) != 0:
                            xadv_npf[:,j,i][defaults_npf] = npf[:,j,i][defaults_npf]

                        if restrict_impact > 0:
                            difference = xadv_npf[:,j,i] - npf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(npf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_npf[high_impact,j,i] = npf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_npf[high_impact,j,i]   

            for j in range(cands_per_variable['vtx']):
                for i in range(vars_per_candidate['vtx']):
                    if i in integer_variables_by_candidate['vtx']:
                        xadv_vtx[:,j,i] = vtx[:,j,i]
                    else:
                        defaults_vtx = vtx[:,j,i].cpu() == defaults_per_variable['vtx'][i]
                        if torch.sum(defaults_vtx) != 0:
                            xadv_vtx[:,j,i][defaults_vtx] = vtx[:,j,i][defaults_vtx]

                        if restrict_impact > 0:
                            difference = xadv_vtx[:,j,i] - vtx[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(vtx[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_vtx[high_impact,j,i] = vtx[high_impact,j,i] + allowed_perturbation[high_impact] * dx_vtx[high_impact,j,i]   

        return xadv_glob.detach(),xadv_cpf.detach(),xadv_npf.detach(),xadv_vtx.detach()
    
def fgsm_attack_domain(epsilon="default",sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, dev=torch.device("cpu"), restrict_impact=-1, epsilon_factors=None, allow_zeros=True):
    glob, cpf, npf, vtx = sample
    y = targets
    xadv_glob = glob.clone().detach()
    xadv_cpf = cpf.clone().detach()
    xadv_npf = npf.clone().detach()
    xadv_vtx = vtx.clone().detach()

    xadv_glob.requires_grad = True
    xadv_cpf.requires_grad = True
    xadv_npf.requires_grad = True
    xadv_vtx.requires_grad = True

    preds = thismodel(xadv_glob,xadv_cpf,xadv_npf,xadv_vtx)
    loss = thiscriterion(preds, targets)

    thismodel.zero_grad()
    loss.backward()
    
    # path to the stored epsilon tensors 
    epsilonpath="/afs/cern.ch/user/h/heschone/private/DeepJet/epsilon_tensors/"+epsilon+"/"

    with torch.no_grad():
        dx_glob = torch.sign(xadv_glob.grad.detach())
        dx_cpf = torch.sign(xadv_cpf.grad.detach())
        dx_npf = torch.sign(xadv_npf.grad.detach())
        dx_vtx = torch.sign(xadv_vtx.grad.detach())
        
        batchsize = y.size(dim=0)
        
        pt_edges = torch.tensor(np.load(epsilonpath+"pt_edges.npy"))
        eta_edges = torch.tensor(np.load(epsilonpath+"eta_edges.npy"))
        epsilons = torch.tensor(np.load(epsilonpath+"epsilons.npy"))
        flavour_bins = [1,2,3]
        
        pt = glob[:,0]
        eta = glob[:,1]
        flavour = flavour_bins[0]*(y[:,0]+y[:,1]+y[:,2]) + flavour_bins[1]*(y[:,3]) + flavour_bins[2]*(y[:,4]+y[:,5])
        pt_tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(pt, dim=1) ,dim=1), dim=1).expand(len(pt), len(flavour_bins) ,len(pt_edges)-1 ,len(eta_edges)-1)
        eta_tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(eta, dim=1),dim=1 ),dim=1 ).expand(len(pt), len(flavour_bins), len(pt_edges)-1, len(eta_edges)-1)
        flavour_tensor = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(flavour, dim=1), dim=1), dim=1).expand(len(pt), len(flavour_bins), len(pt_edges)-1, len(eta_edges)-1)
        
        flavour_filter = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze( torch.tensor(flavour_bins), dim=1), dim=1), dim=0).expand(len(pt), len(flavour_bins), len(pt_edges)-1, len(eta_edges)-1)
        pt_lower = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(pt_edges[:-1], dim=1), dim=0), dim=0 ).expand(len(pt), len(flavour_bins), len(pt_edges)-1, len(eta_edges)-1)
        pt_upper = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(pt_edges[1:], dim=1), dim=0), dim=0 ).expand(len(pt), len(flavour_bins), len(pt_edges)-1, len(eta_edges)-1)
        eta_lower = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(eta_edges[:-1], dim=0), dim=0), dim=0 ).expand(len(pt), len(flavour_bins), len(pt_edges)-1, len(eta_edges)-1)
        eta_upper=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(eta_edges[1:], dim=0), dim=0), dim=0 ).expand(len(pt), len(flavour_bins), len(pt_edges)-1, len(eta_edges)-1)
        
        condition = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and((flavour_tensor==flavour_filter),(pt_tensor>pt_lower)), (pt_tensor<pt_upper)), (eta_tensor>eta_lower)), (eta_tensor<eta_upper))
        domain = torch.where(condition,1,0)
        
        epsilon_tensor=torch.unsqueeze(epsilons, dim=0).expand(len(pt),len(flavour_bins),len(pt_edges)-1,len(eta_edges)-1)
        epsilon_vector=torch.sum(domain*epsilon_tensor,dim=(1,2,3))
        
        if (epsilon_vector==0).nonzero().size(dim=0)>0 and allow_zeros==False:
            zeros = (epsilon_vector==0).nonzero()
            print("For index ",zeros[0].item())
            print("truth label: ",y[zeros[0]])
            print("pt: ",pt[zeros[0]])
            print("eta: ",eta[zeros[0]])
            print("epsilon: ",epsilon_vector[zeros[0]])
            print(condition[zeros[0]])
            raise ValueError("epsilon_vector has zero entry")
        
        # broadcast epsilon into the shapes of glob,cpf,npf,vtx and apply FGSM attack
        xadv_glob+=torch.unsqueeze(epsilon_vector,dim=1).expand(glob.size(dim=0),glob.size(dim=1)) * epsilon_factors['glob'] * dx_glob
        xadv_cpf+=torch.unsqueeze(torch.unsqueeze(epsilon_vector, dim=1),dim=1).expand(cpf.size(dim=0), cpf.size(dim=1), cpf.size(dim=2)) * epsilon_factors['cpf'] * dx_cpf
        xadv_npf+=torch.unsqueeze(torch.unsqueeze(epsilon_vector, dim=1),dim=1).expand(npf.size(dim=0), npf.size(dim=1), npf.size(dim=2)) * epsilon_factors['npf'] * dx_npf
        xadv_vtx+=torch.unsqueeze(torch.unsqueeze(epsilon_vector, dim=1),dim=1).expand(vtx.size(dim=0), vtx.size(dim=1), vtx.size(dim=2)) * epsilon_factors['vtx'] * dx_vtx

        if (glob[:,0]==xadv_glob[:,0]).nonzero().size(dim=0)>0 and allow_zeros==False:
            zeros = (glob[:,0]==xadv_glob[:,0]).nonzero()
            print("For index ",zeros[0].item())
            print("epsilon: ",epsilon_vector[zeros[0]])
            print(dx_glob[zeros[0]-1:zeros[0]+2,:])
            raise ValueError("no change after applying fgsm attack")
        
        if reduced:
            for i in range(vars_per_candidate['glob']):
                if i in integer_variables_by_candidate['glob']:
                    xadv_glob[:,i] = glob[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:,i].cpu() == defaults_per_variable['glob'][i]
                    if torch.sum(defaults_glob) != 0:
                        xadv_glob[:,i][defaults_glob] = glob[:,i][defaults_glob]

                    if restrict_impact > 0:
                        difference = xadv_glob[:,i] - glob[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(glob[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv_glob[high_impact,i] = glob[high_impact,i] + allowed_perturbation[high_impact] * dx_glob[high_impact,i]

            for j in range(cands_per_variable['cpf']):
                for i in range(vars_per_candidate['cpf']):
                    if i in integer_variables_by_candidate['cpf']:
                        xadv_cpf[:,j,i] = cpf[:,j,i]
                    else:
                        defaults_cpf = cpf[:,j,i].cpu() == defaults_per_variable['cpf'][i]
                        if torch.sum(defaults_cpf) != 0:
                            xadv_cpf[:,j,i][defaults_cpf] = cpf[:,j,i][defaults_cpf]

                        if restrict_impact > 0:
                            difference = xadv_cpf[:,j,i] - cpf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(cpf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_cpf[high_impact,j,i] = cpf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_cpf[high_impact,j,i]        

            for j in range(cands_per_variable['npf']):
                for i in range(vars_per_candidate['npf']):
                    if i in integer_variables_by_candidate['npf']:
                        xadv_npf[:,j,i] = npf[:,j,i]
                    else:
                        defaults_npf = npf[:,j,i].cpu() == defaults_per_variable['npf'][i]
                        if torch.sum(defaults_npf) != 0:
                            xadv_npf[:,j,i][defaults_npf] = npf[:,j,i][defaults_npf]

                        if restrict_impact > 0:
                            difference = xadv_npf[:,j,i] - npf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(npf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_npf[high_impact,j,i] = npf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_npf[high_impact,j,i]   

            for j in range(cands_per_variable['vtx']):
                for i in range(vars_per_candidate['vtx']):
                    if i in integer_variables_by_candidate['vtx']:
                        xadv_vtx[:,j,i] = vtx[:,j,i]
                    else:
                        defaults_vtx = vtx[:,j,i].cpu() == defaults_per_variable['vtx'][i]
                        if torch.sum(defaults_vtx) != 0:
                            xadv_vtx[:,j,i][defaults_vtx] = vtx[:,j,i][defaults_vtx]

                        if restrict_impact > 0:
                            difference = xadv_vtx[:,j,i] - vtx[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(vtx[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_vtx[high_impact,j,i] = vtx[high_impact,j,i] + allowed_perturbation[high_impact] * dx_vtx[high_impact,j,i]   
        
        return xadv_glob.detach(),xadv_cpf.detach(),xadv_npf.detach(),xadv_vtx.detach()

# not ready yet
def ngm_attack(epsilon=1e-2,sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, dev=torch.device("cpu"), restrict_impact=-1, epsilon_factors=None):
    if epsilon == 0:
        return sample

    glob, cpf, npf, vtx = sample
    xadv_glob = glob.clone().detach()
    xadv_cpf = cpf.clone().detach()
    xadv_npf = npf.clone().detach()
    xadv_vtx = vtx.clone().detach()

    xadv_glob.requires_grad = True
    xadv_cpf.requires_grad = True
    xadv_npf.requires_grad = True
    xadv_vtx.requires_grad = True

    preds = thismodel(xadv_glob,xadv_cpf,xadv_npf,xadv_vtx)

    loss = thiscriterion(preds, targets)

    thismodel.zero_grad()
    loss.backward()

    with torch.no_grad():
        dx_glob = xadv_glob.grad.detach()
        dx_cpf = xadv_cpf.grad.detach()
        dx_npf = xadv_npf.grad.detach()
        dx_vtx = xadv_vtx.grad.detach()
        
        dx_sum = torch.sum(dx_glob)+torch.sum(dx_cpf)+torch.sum(dx_npf)+torch.sum(dx_vtx)

        xadv_glob += epsilon * epsilon_factors['glob'] * dx_glob / dx_sum
        xadv_cpf += epsilon * epsilon_factors['cpf'] * dx_cpf / dx_sum
        xadv_npf += epsilon * epsilon_factors['npf'] * dx_npf / dx_sum
        xadv_vtx += epsilon * epsilon_factors['vtx'] * dx_vtx / dx_sum

        if reduced:
            for i in range(vars_per_candidate['glob']):
                if i in integer_variables_by_candidate['glob']:
                    xadv_glob[:,i] = glob[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults_glob = glob[:,i].cpu() == defaults_per_variable['glob'][i]
                    if torch.sum(defaults_glob) != 0:
                        xadv_glob[:,i][defaults_glob] = glob[:,i][defaults_glob]

                    if restrict_impact > 0:
                        difference = xadv_glob[:,i] - glob[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(glob[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv_glob[high_impact,i] = glob[high_impact,i] + allowed_perturbation[high_impact] * dx_glob[high_impact,i]

            for j in range(cands_per_variable['cpf']):
                for i in range(vars_per_candidate['cpf']):
                    if i in integer_variables_by_candidate['cpf']:
                        xadv_cpf[:,j,i] = cpf[:,j,i]
                    else:
                        defaults_cpf = cpf[:,j,i].cpu() == defaults_per_variable['cpf'][i]
                        if torch.sum(defaults_cpf) != 0:
                            xadv_cpf[:,j,i][defaults_cpf] = cpf[:,j,i][defaults_cpf]

                        if restrict_impact > 0:
                            difference = xadv_cpf[:,j,i] - cpf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(cpf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_cpf[high_impact,j,i] = cpf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_cpf[high_impact,j,i]        

            for j in range(cands_per_variable['npf']):
                for i in range(vars_per_candidate['npf']):
                    if i in integer_variables_by_candidate['npf']:
                        xadv_npf[:,j,i] = npf[:,j,i]
                    else:
                        defaults_npf = npf[:,j,i].cpu() == defaults_per_variable['npf'][i]
                        if torch.sum(defaults_npf) != 0:
                            xadv_npf[:,j,i][defaults_npf] = npf[:,j,i][defaults_npf]

                        if restrict_impact > 0:
                            difference = xadv_npf[:,j,i] - npf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(npf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_npf[high_impact,j,i] = npf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_npf[high_impact,j,i]   

            for j in range(cands_per_variable['vtx']):
                for i in range(vars_per_candidate['vtx']):
                    if i in integer_variables_by_candidate['vtx']:
                        xadv_vtx[:,j,i] = vtx[:,j,i]
                    else:
                        defaults_vtx = vtx[:,j,i].cpu() == defaults_per_variable['vtx'][i]
                        if torch.sum(defaults_vtx) != 0:
                            xadv_vtx[:,j,i][defaults_vtx] = vtx[:,j,i][defaults_vtx]

                        if restrict_impact > 0:
                            difference = xadv_vtx[:,j,i] - vtx[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(vtx[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_vtx[high_impact,j,i] = vtx[high_impact,j,i] + allowed_perturbation[high_impact] * dx_vtx[high_impact,j,i]   

        return xadv_glob.detach(),xadv_cpf.detach(),xadv_npf.detach(),xadv_vtx.detach()