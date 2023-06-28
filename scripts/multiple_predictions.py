# Hendrik Schönen
# This script calls the script predict_pytorch.py multiple times to make predictions for multiple models and attacks

import subprocess

model_names = ['nominal','fgsm-0_01']
predictions = ['fgsm-0_1']#'nominal','fgsm-0_01','fgsm-0_02','fgsm-0_05',
restrict_impact = 0.2
checkpoint = 'checkpoint_best_loss.pth'
model_dir = '/net/scratch_cms3a/hschoenen/deepjet/results/'


prediction_commands={
    'nominal': '/predict_nominal',
    'fgsm-0_01': '/predict_fgsm-0_01 -attack FGSM -att_magnitude 0.01',
    'fgsm-0_02': '/predict_fgsm-0_02 -attack FGSM -att_magnitude 0.02',
    'fgsm-0_05': '/predict_fgsm-0_05 -attack FGSM -att_magnitude 0.05',
    'fgsm-0_1': '/predict_fgsm-0_1 -attack FGSM -att_magnitude 0.1',
    'fgsm_flavour-0_01-0_01-0_01': '/predict_fgsm_flavour_default -attack FGSM_flavour -att_magnitude_flavour 0.01 0.01 0.01',
    'fgsm_domain_default': '/predict_fgsm_domain_default -attack FGSM_domain -att_magnitude_domain default',
}

for model in model_names:
    for prediction in predictions:
        command='python3 pytorch/predict_pytorch.py DeepJet_Run2 '+ model_dir+model+'/'+ checkpoint + ' ' + model_dir+ model+'/trainsamples.djcdc' + ' one_sample_lxportal.txt ' + model_dir+model+prediction_commands[prediction] + ' -restrict_impact '+str(restrict_impact)
        print(model+"    "+prediction)
        print(command)
        p=subprocess.run(command.split(" "))
        
# Alternative (with no QCD) to onesample.txt: /eos/cms/store/group/phys_btag/ParticleTransformer/ntuple_ttbar_had_test_samples/output/samples.txt