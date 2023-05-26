#!/bin/bash

echo "Setting up container."

source $HOME/private/DeepJetCore/docker_env.sh
cd $HOME/private/DeepJet
source env.sh

export PYTHONPATH=/net/scratch_cms3a/hschoenen/python3.6.8/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=/net/scratch_cms3a/hschoenen/python3.6.8/lib64/python3.6/site-packages:$PYTHONPATH
export PATH=/net/scratch_cms3a/hschoenen/python3.6.8/bin:$PATH

source lxportal/execute_payload.sh