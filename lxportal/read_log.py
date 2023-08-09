import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Read the latest condor logs.')
parser.add_argument("-i", help="index of the logfile in logfiles.txt",default="-1")
args = parser.parse_args()
i = int(args.i)

logfile_list = '/home/home1/institut_3a/hschoenen//repositories/DeepJet/lxportal/logfiles.txt'
file = open(logfile_list, "r")

logfiles = []
for logfile in file:
    logfiles.append(logfile[:-1])

print('read logfile: {}'.format(logfiles[i]))
os.system('tail -f {}'.format(logfiles[i]))