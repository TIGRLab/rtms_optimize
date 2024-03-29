#!/usr/bin/env python
# coding: utf-8

import os
import gc

import numpy as np
from backports import tempfile #python 2 workaround for lack of context in tempfile
from fieldopt import geolib
from simnibs import sim_struct, run_simulation
from functools import wraps

class FieldFunc():

    FIELD_ENTITY = (3,2)

    '''
    This class provides an interface in which the details related to simulation and score extraction
    is abstracted away for any general Bayesian Optimization package

    Layout
    Properties:
        1. Mesh file to perform simulation on
        2. Store details surrounding the input --> surface transformations (use file paths?)

    '''

    def __init__(self, mesh_file, quad_surf_consts,
            surf_to_mesh_matrix, tet_weights, field_dir, coil,
            cpus=1):
        '''
        Standard constructor
        Arguments:
            mesh_file                   Path to FEM model
            quad_surf_consts            Quadratic surface constants
            surf_to_mesh_matrix         (3,3) affine transformation matrix
            tet_weights                 Weighting scores for each tetrahedron (1D array ordered by node ID)
            field_dir                   Directory to perform simulation experiments in
            coil                        TMS coil file (either dA/dt volume or coil geometry)
        '''

        self.mesh = mesh_file
        self.C = quad_surf_consts
        self.iR = surf_to_mesh_matrix
        self.tw = tet_weights
        self.field_dir = field_dir
        self.coil = coil
        self.cpus = cpus


    def __repr__(self):
        '''
        print(FieldFunc)
        '''

        print('Mesh:', self.mesh)
        print('Coil:', self.coil)
        print('Field Directory:', self.field_dir)
        return ''

    def _transform_input(self, x, y, theta):
        '''
        Generates a coil orientation matrix given inputs from a
        quadratic surface sampling domain
        '''

        preaff_loc = geolib.map_param_2_surf(x,y,self.C)
        preaff_rot, preaff_norm = geolib.map_rot_2_surf(x, y, theta, self.C)

        loc = np.matmul(self.iR, preaff_loc)
        rot = np.matmul(self.iR, preaff_rot)
        n = np.matmul(self.iR, preaff_norm)

        o_matrix = geolib.define_coil_orientation(loc, rot, n)
        return o_matrix

    def _run_simulation(self, matsimnibs, sim_dir):
        '''
        Arguments:
            matsimnibs                  Coil orientation matrix (simnibs specification)
            sim_dir                     Directory to perform simulation experiment
        '''

        S = sim_struct.SESSION()
        S.fnamehead = self.mesh
        S.pathfem = sim_dir

        tms = S.add_tmslist()
        tms.fnamecoil = self.coil

        for i in np.arange(0,len(matsimnibs)):
            p = sim_struct.POSITION()
            p.matsimnibs = matsimnibs[i]
            tms.add_position(p)

        sim_files = S.run_simulatons(cpus=self.cpus)

        #Needed since simNIBS is bad about circular referencing
        del S
        del tms
        gc.collect()
        return sorted(sim_files)

    def _calculate_score(self, sim_file):
        '''
        Given a simulation output file, compute the score
        '''

        _, elem_ids, _ = geolib.load_gmsh_elems(sim_file, self.FIELD_ENTITY)
        normE = geolib.get_field_subset(sim_file, elem_ids)
        return np.dot(self.tw, normE)

    def evaluate(self, input_list):

        '''
        Given a quadratic surface input (x,y) and rotational interpolation angle (theta)
        compute the resulting field score over a region of interest
        Arguments:
            [(x,y,theta),...]                   A iterable of iterable (x,y,theta)

        Returns:
            scores                              An array of scores in order of inputs
        '''

        with tempfile.TemporaryDirectory(dir=self.field_dir) as sim_dir:

            matsimnibs = [self._transform_input(x,y,t) for x,y,t in input_list]
            sim_files = self._run_simulation(matsimnibs, sim_dir)
            scores = np.array([self._calculate_score(s) for s in sim_files])

        return scores
