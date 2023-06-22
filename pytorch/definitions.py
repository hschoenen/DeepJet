cands_per_variable = {
    'glob' : 1,
    'cpf' : 25,
    'npf' : 25,
    'vtx' : 4,
}
vars_per_candidate = {
    'glob' : 15,
    'cpf' : 16,
    'npf' : 6,
    'vtx' : 12,
}
defaults_per_variable_before_prepro = {
    'glob' : [None,None,None,None,None,None,-999,-999,-999,-999,-999,-999,-999,None,None],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
}
epsilons_per_feature = {
    'glob' : '/home/home1/institut_3a/hschoenen/repositories/DeepJet/epsilons/global_standardized_epsilons.npy',
    'cpf' : '/home/home1/institut_3a/hschoenen/repositories/DeepJet/epsilons/cpf_standardized_epsilons.npy',
    'npf' : '/home/home1/institut_3a/hschoenen/repositories/DeepJet/epsilons/npf_standardized_epsilons.npy',
    'vtx' : '/home/home1/institut_3a/hschoenen/repositories/DeepJet/epsilons/vtx_standardized_epsilons.npy',
}
defaults_per_variable = {
    'glob' : [0 for i in range(vars_per_candidate['glob'])],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
}
integer_variables_by_candidate = {
    'glob' : [2,3,4,5,8,13,14],
    'cpf' : [12,13,14,15], # adding 14 because chi2 is an approximante integer
    'npf' : [2],
    'vtx' : [3],
}
