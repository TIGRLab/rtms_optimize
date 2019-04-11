#!/usr/bin/env python

import gmsh
import os
import numpy as np
from fieldopt import geolib
from simnibs.msh import gmsh_numpy as simgmsh
import pdb
import objgraph

proj_dir = '/projects/jjeyachandra/rtms_optimize'
local_dir=os.path.join(proj_dir,'data','simnibs_output')
fnamehead = os.path.join(proj_dir,'data','simnibs_output','sub-CMH090.msh')

n_iter = 1
num_parcels = 20
target_parcel = 2
        
i=0
pathfem = os.path.join(proj_dir,'objective_grid','grid','sim_{}'.format(i+3000))
field_file=os.path.join(pathfem,'sub-CMH090_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh')

objgraph.show_most_common_types(limit=50)
for i in np.arange(0,100):
    msh = simgmsh.read_msh(field_file)

objgraph.show_most_common_types(limit=50)
pdb.set_trace()


