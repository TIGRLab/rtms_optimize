#!/usr/bin/env python

import os
import numpy as np
from fieldopt import geolib
from fieldopt.objective import FieldFunc

from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cTensorProductDomain
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import multistart_expected_improvement_optimization as meio
from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import GaussianProcessLogLikelihoodMCMC
from moe.optimal_learning.python.default_priors import DefaultPrior
from moe.optimal_learning.python.python_version.optimization import GradientDescentOptimizer, GradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cGDOpt
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cGDParams

#Specify inputs
proj_dir =  '/projects/jjeyachandra/rtms_optimize'
mesh_file = os.path.join(proj_dir,'data','simnibs_output','sub-CMH090.msh')
coil_file = os.path.join(proj_dir,'resources','coils','Magstim_70mm_Fig8.nii.gz')
tet_file = os.path.join(proj_dir,'output','tetra_parcels')
C_file = os.path.join(proj_dir,'output','quadratic_vec')
iR_file = os.path.join(proj_dir,'output','inverse_rot')
b_file = os.path.join(proj_dir,'output','param_bounds')
testing_dir = '/tmp/'#os.path.join(proj_dir,'testing','tmp')

#Step 1: Load in files containing relevant information about search domain
C = np.fromfile(C_file)
iR = np.fromfile(iR_file).reshape(3,3)
b = np.fromfile(b_file).reshape(3,2)

#Load in element mask
p_map = np.load(os.path.join(proj_dir,'output','tetra_parcels.npy'))[:,2]

#Step 2: Specify the search domain
search_domain = TensorProductDomain([
        ClosedInterval(b[0,0],b[0,1]), #X coord on quadratic surface
            ClosedInterval(b[1,0],b[1,1]), #Y coord on quadratic surface
                ClosedInterval(0,180) #Rotational angle
                ])

c_search_domain = cTensorProductDomain([
        ClosedInterval(b[0,0],b[0,1]),
            ClosedInterval(b[1,0],b[1,1]),
                ClosedInterval(0,180)
        ])

#Try to sample and point and view outcome
x,y,t = search_domain.generate_random_point_in_domain()

#Make objective function
f = FieldFunc(mesh_file=mesh_file, quad_surf_consts=C,
    surf_to_mesh_matrix=iR, tet_weights=p_map,
    field_dir=testing_dir, coil=coil_file)
score = f.evaluate(x,y,t)
print(score)

#Sample again
f.evaluate(x,y,t)
