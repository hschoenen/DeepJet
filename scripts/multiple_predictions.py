# Hendrik Schönen
# This script calls the script predict_pytorch.py multiple times to make predictions for multiple models and attacks

import subprocess

model_names = ['nominal','nominal_2','nominal_3','nominal_seed1','nominal_seed2']#'nominal','seed_nominal','seed_nominal2','seed_nominal3']#,'fgsm-0_025','fgsm-0_05','fgsm-0_075','fgsm-0_1','fgsm-0_125','fgsm-0_15','fgsm-0_175','fgsm-0_2','fgsm-0_225','fgsm-0_25']#
predictions = ['nominal','fgsm-0_1']#'gaussian-0_1','gaussian-1','gaussian-5']#'nominal','fgsm-0_025','fgsm-0_05','fgsm-0_075','fgsm-0_1','fgsm-0_125','fgsm-0_15','fgsm-0_175','fgsm-0_2','fgsm-0_225','fgsm-0_25'
restrict_impact = 0.2
checkpoint = 'checkpoint_best_loss.pth'
#model_dir = '/net/scratch_cms3a/hschoenen/deepjet/results/'
model_dir = '/net/data_cms/institut_3a/hschoenen/models/'


prediction_commands={
    'nominal': '/predict_nominal',
    'fgsm-0_01': '/predict_fgsm-0_01 -attack FGSM -att_magnitude 0.01',
    'fgsm-0_025': '/predict_fgsm-0_025 -attack FGSM -att_magnitude 0.025',
    'fgsm-0_05': '/predict_fgsm-0_05 -attack FGSM -att_magnitude 0.05',
    'fgsm-0_075': '/predict_fgsm-0_075 -attack FGSM -att_magnitude 0.075',
    'fgsm-0_1': '/predict_fgsm-0_1 -attack FGSM -att_magnitude 0.1',
    'fgsm-0_125': '/predict_fgsm-0_125 -attack FGSM -att_magnitude 0.125',
    'fgsm-0_15': '/predict_fgsm-0_15 -attack FGSM -att_magnitude 0.15',
    'fgsm-0_175': '/predict_fgsm-0_175 -attack FGSM -att_magnitude 0.175',
    'fgsm-0_2': '/predict_fgsm-0_2 -attack FGSM -att_magnitude 0.2',
    'fgsm-0_225': '/predict_fgsm-0_225 -attack FGSM -att_magnitude 0.225',
    'fgsm-0_25': '/predict_fgsm-0_25 -attack FGSM -att_magnitude 0.25',
    'fgsm_flavour-0_01-0_01-0_01': '/predict_fgsm_flavour_default -attack FGSM_flavour -att_magnitude_flavour 0.01 0.01 0.01',
    'fgsm_domain_default': '/predict_fgsm_domain_default -attack FGSM_domain -att_magnitude_domain default',
    'gaussian-0_05': '/predict_gaussian-0_05 -attack Gaussian -att_magnitude 0.05',
    'gaussian-0_1': '/predict_gaussian-0_1 -attack Gaussian -att_magnitude 0.1',
    'gaussian-0_2': '/predict_gaussian-0_2 -attack Gaussian -att_magnitude 0.2',
    'gaussian-1': '/predict_gaussian-1 -attack Gaussian -att_magnitude 1.0',
    'gaussian-5': '/predict_gaussian-5 -attack Gaussian -att_magnitude 5.0',
    
}

for model in model_names:
    for prediction in predictions:
        command='python3 pytorch/predict_pytorch.py DeepJet_Run2 '+ model_dir+model+'/'+ checkpoint + ' ' + model_dir+ model+'/trainsamples.djcdc' + ' one_sample_lxportal.txt ' + model_dir+model+prediction_commands[prediction] + ' -restrict_impact '+str(restrict_impact)
        print(model+"    "+prediction)
        print(command)
        p=subprocess.run(command.split(" "))
        
# Alternative (with no QCD) to onesample.txt: /eos/cms/store/group/phys_btag/ParticleTransformer/ntuple_ttbar_had_test_samples/output/samples.txt