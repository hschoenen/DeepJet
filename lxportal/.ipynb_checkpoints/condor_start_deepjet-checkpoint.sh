#!/bin/bash

apptainer exec --nv --bind=/home/home1/institut_3a/hschoenen/,/net /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest /bin/bash /home/home1/institut_3a/hschoenen/repositories/DeepJet/lxportal/condor_setup_container.sh
