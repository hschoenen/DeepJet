import numpy as np
import torch
import os

print(f"This process has the PID {os.getpid()} .")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = ["nominal", "fgsm"]
adv    = [false, True]


def save_loss(model_names, is_adv):
    base_dir = "/net/scratch_cms3a/hschoenen/deepjet/results/"
    
    for i,name in enumerate(model_names):
        if is_adv[i]:
            num_epochs = np.arange(1,79)
        else:
            num_epochs = np.arange(1,40)
            
        model_dirs = [base_dir + name + f"checkpoint_epoch_{epoch}.pth" for epoch in num_epochs]
        
        train_loss = []
        val_loss   = []
        for model in model_dirs:
            checkpoint = torch.load(model, map_location=torch.device(device))
            train_loss.append(checkpoint["train_loss"])
            val_loss.append(checkpoint["val_loss"])
        np.save(base_dir + name + f"loss.npy", np.array([train_loss,val_loss]))

save_loss(models, adv)

print(f"Finished process {os.getpid()} .")