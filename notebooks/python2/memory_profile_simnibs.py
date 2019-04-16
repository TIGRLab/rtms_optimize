#!/usr/bin/env python

import os
import gc
import objgraph
from simnibs.msh import gmsh_numpy as simgmsh

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
for i in np.arange(0,20):

    #GC.COLLECT() needed to deal with circular referencing in Python API for simnibs
    #gc.collect()
    msh = simgmsh.read_msh(field_file)

objgraph.show_most_common_types(limit=50)
obj = objgraph.by_type('Msh')[0]
objgraph.show_backrefs(obj, max_depth=10, filename='./Msh_backref.png')


