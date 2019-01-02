#!/usr/bin/env python

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Take 3x3 subset of rotation matrix and normalize scaling component via singular value decomposition. In addition set translational component to identity.")
parser.add_argument(dest='mat', metavar='M', help="4x4 Matrix File")
parser.add_argument(dest='out', metavar='out_path', type=str, help="Output file path")
args = parser.parse_args()

#Load in text file
M = np.genfromtxt(args.mat)

#Set translational component to identity
M[:,3] = 0
M[3,3] = 1

#Normalize the linear component
linear_map = M[:3,:3]
U,S,V = np.linalg.svd(linear_map)
M[:3,:3] = np.matmul(U,V)

#Write normalized/linearized transformation
np.savetxt(args.out,M)
