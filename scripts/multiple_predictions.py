# Hendrik Schönen
# This script calls the script predict_pytorch.py multiple times to make predictions for multiple models and attacks

import subprocess

model_names=['fgsm']#,'fgsm_flavour_006008010','fgsm_flavour_008009010','fgsm_flavour_010009010','nominal']
predictions=['nominal','fgsm','fgsm_flavour_012008010','fgsm_domain_random'] #'fgsm_flavour_default','fgsm_flavour_006008010','fgsm_flavour_008009010','fgsm_flavour_010009010','fgsm_domain_default','NGM''fgsm_domain_default',]

checkpoint='checkpoint_best_loss.pth'
prediction_commands={
    'nominal': '/predict',
    'fgsm': '/predict_FGSM -attack FGSM -att_magnitude 0.01 -restrict_impact 0.2',
    'fgsm_flavour_default': '/predict_FGSM_flavour_default -attack FGSM_flavour -att_magnitude_flavour 0.01 0.01 0.01 -restrict_impact 0.2',
    'fgsm_flavour_006008010': '/predict_FGSM_flavour_006008010 -attack FGSM_flavour -att_magnitude_flavour 0.006 0.008 0.01 -restrict_impact 0.2',
    'fgsm_flavour_008009010': '/predict_FGSM_flavour_008009010 -attack FGSM_flavour -att_magnitude_flavour 0.008 0.009 0.01 -restrict_impact 0.2',
    'fgsm_flavour_010009010': '/predict_FGSM_flavour_010009010 -attack FGSM_flavour -att_magnitude_flavour 0.01 0.009 0.01 -restrict_impact 0.2',
    'fgsm_flavour_012008010': '/predict_FGSM_flavour_012008010 -attack FGSM_flavour -att_magnitude_flavour 0.012 0.008 0.01 -restrict_impact 0.2',
    'fgsm_domain_default': '/predict_FGSM_domain_default -attack FGSM_domain -att_magnitude default -restrict_impact 0.2',
    'fgsm_domain_random': '/predict_FGSM_domain_random -attack FGSM_domain -att_magnitude random -restrict_impact 0.2',
    'NGM': '/predict_NGM -attack NGM -att_magnitude 0.01 -restrict_impact 0.2',
}

for model in model_names:
    for prediction in predictions:
        command='python3 pytorch/predict_pytorch.py DeepJet_Run2 /eos/user/h/heschone/DeepJet/Train_DF_Run2/'+ model + '/' + checkpoint + ' /eos/user/h/heschone/DeepJet/Train_DF_Run2/' + model + '/trainsamples.djcdc one_sample.txt /eos/user/h/heschone/DeepJet/Train_DF_Run2/' + model + prediction_commands[prediction] 
        #print(command)
        print(model+"    "+prediction)
        p=subprocess.run(command.split(" "))
        
# Alternative (with no QCD) to onesample.txt: /eos/cms/store/group/phys_btag/ParticleTransformer/ntuple_ttbar_had_test_samples/output/samples.txt