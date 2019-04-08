#!/usr/bin/env python

#SBATCH --partition=high-moby
#SBATCH --output=/projects/jjeyachandra/rtms_optimize/testing/testing_%a.out
#SBATCH --error=/projects/jjeyachandra/rtms_optimize/testing/testing_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --job-name=testing_jupyter_slurm
#SBATCH --array=0-1

#Package import
import sys
sys.path.insert(0,'/projects/jjeyachandra/simnibs/miniconda2/envs/simnibs_env/lib/python2.7/site-packages')
sys.path.insert(0,'/projects/jjeyachandra/simnibs/Python_modules/src')

import os
from simnibs import sim_struct, run_simulation
from fieldopt import geolib, tetrapro
import numpy as np

####### DATA LOADING

#File specification
proj_dir = '/projects/jjeyachandra/rtms_optimize'
C = np.fromfile(os.path.join(proj_dir,'output','quadratic_vec'))
iR = np.fromfile(os.path.join(proj_dir,'output','inverse_rot')).reshape(3,3)
b = np.fromfile(os.path.join(proj_dir,'output','param_bounds')).reshape(3,2)

###### INPUT SETUP 

#Set up evaluation grid
fidelity = 20
x_samps = np.linspace(b[0,0],b[0,1],fidelity)
y_samps = np.linspace(b[1,0],b[1,1],fidelity)
rot_samps = np.linspace(0,180,fidelity)
XX,YY,RR = np.meshgrid(x_samps,y_samps,rot_samps)
XX = XX.flatten().reshape((fidelity**3,1))
YY = YY.flatten().reshape((fidelity**3,1))
RR = RR.flatten().reshape((fidelity**3,1))
sample_set = np.concatenate((XX,YY,RR),axis=1)

#Pick input parameters
ind = 0#int(os.environ['SLURM_ARRAY_TASK_ID'])
x,y,r = sample_set[ind,:] 

#Set up spatial postioning
preaff_loc = geolib.map_param_2_surf(x,y,C)
loc = np.matmul(iR,preaff_loc)

#Set up rotational positioning
preaff_rot, preaff_n = geolib.map_rot_2_surf(x,y,r,C)
rot = np.matmul(iR,preaff_rot)
n = np.matmul(iR,preaff_n)

###### SIMULATION SETUP

#Start a session
S = sim_struct.SESSION()
S.fnamehead = os.path.join(proj_dir,'data','simnibs_output','sub-CMH090.msh')
S.pathfem = os.path.join(proj_dir, 'objective_grid','sim_{}'.format(ind))
tms = S.add_tmslist()
tms.fnamecoil='/projects/jjeyachandra/simnibs/ccd-files/Magstim_70mm_Fig8.nii.gz'
pos = tms.add_position()
pos.matsimnibs = geolib.define_coil_orientation(loc,rot,n)

#Run simulation
run_simulation(S)

###### PROCESS OUTPUTS

#Compute score
#Save positional .geo file
