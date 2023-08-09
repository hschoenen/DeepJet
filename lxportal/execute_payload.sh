#!/bin/bash

echo "Executing payload."
echo "############################## END OF BASH LOG ##############################"

### training
#export input_dir=/net/scratch_cms3a/hschoenen/deepjet/data/Data/dataCollection.djcdc
#export output_dir=/net/scratch_cms3a/hschoenen/deepjet/results/fgsm-0_15-updated_epsilons
#export adv=_fgsm
#python3 /home/home1/institut_3a/hschoenen/repositories/DeepJet/pytorch/train_DeepFlavour$adv.py $input_dir $output_dir

### prediction
#python3 /home/home1/institut_3a/hschoenen/repositories/DeepJet/scripts/multiple_predictions.py

### ROC curves
#python3 /home/home1/institut_3a/hschoenen/repositories/DeepJet/scripts/plot_roc.py

### plot inputs
#python3 pytorch/plot_inputs.py

### plot loss surfaces
python3 pytorch/plot_loss_surface.py

### testing
#echo; export; echo; nvidia-smi; echo; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvcc --version
#python3 $HOME/wor