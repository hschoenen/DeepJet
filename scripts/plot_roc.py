# Hendrik Schönen
# This script produces datapoints for BvsL, CvsB and CvsL ROC curves
# It requires prediction files produced by pytorch/predict_pytorch.py

from sklearn.metrics import roc_curve, auc
import numpy.lib.recfunctions as rf
import pandas as pd
import numpy as np
import os

print(f"This process has the PID {os.getpid()} .")

models = ["nominal","fgsm-0_01"]#[]"fgsm-0_02"
attacks = ["nominal","fgsm-0_01","fgsm-0_02","fgsm-0_05","fgsm-0_1"]#"fgsm-0_005",,"fgsm-0_015","fgsm-0_02","fgsm--0_01"
predictions = []
for model in models:
    for attack in attacks:
        prediction = model + "/predict_" + attack +"/"
        predictions.append(prediction)

# compute ROC curve for the discriminator values disc, the labels truth_array
def spit_out_roc(disc,truth_array,selection_array):
        tprs                = pd.DataFrame()
        truth               = truth_array[selection_array] * 1
        disc                = disc[selection_array]
        tmp_fpr, tmp_tpr, _ = roc_curve(truth, disc)
        coords              = pd.DataFrame()
        coords["fpr"]       = tmp_fpr
        coords["tpr"]       = tmp_tpr
        clean               = coords.drop_duplicates(subset=["fpr"])
        auc_                = auc(clean.fpr, clean.tpr)
        print("AUC: ", str(auc_))
        print("\n")
        return clean.tpr, clean.fpr, auc_ * np.ones(np.shape(clean.tpr))

def save_roc(prediction_path):
    base_dir         = "/net/scratch_cms3a/hschoenen/deepjet/results/"
    output_dirs      = [base_dir + f"{i}" for i in prediction_path]
    
    listbranch = ["prob_isB", "prob_isBB", "prob_isLeptB", "prob_isC", "prob_isUDS", "prob_isG", "isB", "isBB", "isLeptB", "isC", "isUDS", "isG", "jet_pt", "jet_eta"]

    for j,output in enumerate(output_dirs):
        # load prediction file
        nparray  = rf.structured_to_unstructured(np.array(np.load(output + "pred_ntuple_merged_342.npy")))
        df       = np.core.records.fromarrays([nparray[:,k] for k in range(len(listbranch))], names=listbranch)

        # convert b,bb,lepb,c,uds,g in B,C,L
        b_jets    = df["isB"]        + df["isBB"]      + df["isLeptB"]
        c_jets    = df["isC"]
        b_out     = df["prob_isB"]   + df["prob_isBB"] + df["prob_isLeptB"]
        c_out     = df["prob_isC"]
        light_out = df["prob_isUDS"] + df["prob_isG"]
        
        # compute discriminator values
        bvsl = np.where((b_out + light_out)!=0, (b_out)/(b_out + light_out), -1)
        cvsb = np.where((b_out + c_out)    !=0, (c_out)/(b_out + c_out), -1)
        cvsl = np.where((light_out + c_out)!=0, (c_out)/(light_out + c_out), -1)

        # apply selection (pT>30GeV) and sort jets by true flavor: veto_b=True for NON-b jets, veto_b=False for b jets
        summed_truth = df["isB"] + df["isBB"] + df["isLeptB"] + df["isC"] + df["isUDS"] + df["isG"]
        veto_b    = (df["isB"] != 1)   & (df["isBB"] != 1)    & (df["isLeptB"] != 1) & ( df["jet_pt"] > 30) & (summed_truth != 0)
        veto_c    = (df["isC"] != 1)   & ( df["jet_pt"] > 30) & (summed_truth != 0)
        veto_udsg = (df["isUDS"] != 1) & (df["isG"] != 1)     & ( df["jet_pt"] > 30) & (summed_truth != 0)
        
        # compute ROC curves
        x1, y1, auc1 = spit_out_roc(bvsl,b_jets,veto_c)
        x2, y2, auc2 = spit_out_roc(cvsb,c_jets,veto_udsg)
        x3, y3, auc3 = spit_out_roc(cvsl,c_jets,veto_b)

        # save ROC curves
        np.save(output + "BvL.npy", np.stack((x1, y1, auc1)))
        np.save(output + "CvB.npy", np.stack((x2, y2, auc2)))
        np.save(output + "CvL.npy", np.stack((x3, y3, auc3)))
        
save_roc(predictions)
print(f"Finished process {os.getpid()} .")