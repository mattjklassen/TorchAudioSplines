# ----- Brief Description -----
#
# Plot cubic spline f(t) with bcoeffs coming from file as arg1 on command line, and
# optionally knot sequence from file as arg2.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
#
# Use default knot sequence 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1 unless second
# command line arg2 gives other knot sequence.  From knot sequence of length N+1,
# the dimension of cubic splines is N-3 = n which needs to match the number of bcoeffs.
# example: For the cubic "splinusoid" we match sin(2Pix) on [0,1] at 9 points which are
# endpoints in a uniform partition into 8 subintervals.  Doing half of this problem
# first, on [0,1] approximate sin(Pi*x) at 5 points, is done nicely with knot sequence:
# 0,0,0,0,1/4,1/2,3/4,1,1,1,1.  Gluing this into the 
#
# ----- ----- ----- ----- -----

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from getBcoeffs import import_bcoeffs, export_bcoeffs

# The next two functions are imported to compute using the deBoor algorithm.  The first one computes
# the value of a single B-spline basis function for value t in [0,1].  This is necessary in order to
# set up the linear system to solve for bcoeffs (B-spline coefficients).  The second one computes a 
# value for B-spline function given all n=k+d bcoeffs.  

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 
from computeBsplineVal import computeSplineVal2 

file = "bcoeffs-sin2.txt"
bcoeffs = import_bcoeffs(file)
# for this set of bcoeffs we need knot sequence: 0,0,0,0,1/8,2/8,3/8,4/8,4/8,4/8,4/8,5/8,6/8,7/8,1,1,1,1
# with N = 17.  Then dimension of cubic splines with this knot sequence is n = d+1+3+4+1 = 14.
# (dim n = d+1 + sum of multiplicities m_i at each breakpoint = 4 + 1+1+1+4+1+1+1 = 14.)
n = len(bcoeffs)  # n = 14
d = 3
# k is number of subintervals [0,1/8,2/8,...,7/8,1] so k = 8
k = 8   # k = 8 can be computed from breakpoint sequence
# N is last index of knot sequence of length N+1
N = n + d   # N = 17
print("k, d, n, N:  ", k, d, n, N)

knotVals = torch.zeros(N+1)
inputVals = torch.zeros(9)

incr = 1 / k
for i in range(k+1) :
    inputVals[i] = i * incr
print("inputVals:  ", inputVals)

for i in range(4) :
    j = i + 7
    knotVals[j] = 1/2
    j = i + 14
    knotVals[j] = 1

for i in range(3) :
    j = i + 4
    knotVals[j] = knotVals[j-1] + incr
    j = i + 11
    knotVals[j] = knotVals[j-1] + incr

print("knotVals: ")
print(knotVals)

outputVals = torch.zeros(9)
outputVals[8] = 0
# we will plot points (inputVals, outputVals) with spline plot
for i in range(k+1) :
    t = inputVals[i]
    if t < 1 :
        outputVals[i] = computeSplineVal2(d, bcoeffs, knotVals, t)

# Next graph spline function with the computed coefficients using 1000 points.

xvals = np.arange(start=0.0, stop=1.001, step=.001)
print(xvals)
print("size of xvals:  ", xvals.size)
yvals = np.zeros(1001)
for i in range(1001) :
    t = xvals[i]
    yvals[i] = computeSplineVal2(d, bcoeffs, knotVals, t)

print(yvals)
print("size of yvals:  ", yvals.size)
plt.figure(figsize=(15,8))
plt.plot(xvals, yvals)
plt.plot(inputVals, outputVals, 'ro')
title = "cubic spline matching sin(2Pi*x) on [0,1]"
# plt.title('cubic interpolating spline n=12, start and end at zero, 10 random values in [-1,1]')
plt.title(title)
plt.xlabel('time axis')
# plt.ylabel('audio sample axis')
plt.show()

