#!/bin/bash

echo "Setting up container."


cd /home/home1/institut_3a/hschoenen/repositories/DeepJetCore
source docker_env.sh
cd /home/home1/institut_3a/hschoenen/repositories/DeepJet
source env.sh

export PYTHONPATH=/net/scratch_cms3a/ajung/python3.6.8/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=/net/scratch_cms3a/ajung/python3.6.8/lib64/python3.6/site-packages:$PYTHONPATH
export PATH=/net/scratch_cms3a/ajung/python3.6.8/bin:$PATH

source  /home/home1/institut_3a/hschoenen/repositories/DeepJet/lxportal/execute_payload.sh
 