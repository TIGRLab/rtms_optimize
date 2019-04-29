#!/usr/bin/env python
#SBATCH --partition=high-moby
#SBATCH --output=/projects/jjeyachandra/rtms_optimize/objective_grid/log/sim_%a.out
#SBATCH --error=/projects/jjeyachandra/rtms_optimize/objective_grid/log/sim_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=field_simulations
#SBATCH --array=0-26999%80
#SBATCH --time=0:30:00

import sys
import os
import numpy as np
from fieldopt import geolib
from fieldopt.objective import FieldFunc

#Directory specification
proj_dir =  '/projects/jjeyachandra/rtms_optimize'
mesh_file = os.path.join(proj_dir,'data','simnibs_output','sub-CMH090.msh')
coil_file = os.path.join(proj_dir,'resources','coils','Magstim_70mm_Fig8.nii.gz')
tet_file = os.path.join(proj_dir,'output','tetra_parcels')
C_file = os.path.join(proj_dir,'output','quadratic_vec')
iR_file = os.path.join(proj_dir,'output','inverse_rot')
b_file = os.path.join(proj_dir,'output','param_bounds')
output_dir = os.path.join(proj_dir,'objective','scores')

C = np.fromfile(C_file)
iR = np.fromfile(iR_file).reshape(3,3)
b = np.fromfile(b_file).reshape(3,2)

#Load in element mask
p_map = np.load(os.path.join(proj_dir,'output','tetra_parcels.npy'))[:,2]

f = FieldFunc(mesh_file=mesh_file, quad_surf_consts=C,
                      surf_to_mesh_matrix=iR, tet_weights=p_map,
                                    field_dir='/tmp/', coil=coil_file, cpus=6)

#Set up evaluation grid
fidelity = 30
x_samps = np.linspace(b[0,0],b[0,1],fidelity)
y_samps = np.linspace(b[1,0],b[1,1],fidelity)
rot_samps = np.linspace(0,180,fidelity)
XX,YY,RR = np.meshgrid(x_samps,y_samps,rot_samps)
XX = XX.flatten().reshape((fidelity**3,1))
YY = YY.flatten().reshape((fidelity**3,1))
RR = RR.flatten().reshape((fidelity**3,1))
sample_set = np.concatenate((XX,YY,RR),axis=1)
ind = int(os.environ['SLURM_ARRAY_TASK_ID'])

#Evaluate to extract score
score = f.evaluate([sample_set[ind]])
score_dir=os.path.join(proj_dir,'objective_grid','scores')
score_file=os.path.join(score_dir,'fx_{}'.format(ind))
np.save(score_file,score)

print('Successfully finished simulation with following parameters:')
print('INPUTS: {} {} {}'.format(sample_set[0][0],sample_set[0][1],sample_set[0][2]))
print('INDEX: {}'.format(ind))
print('SCORE:',score)
print('Finished successfully')
