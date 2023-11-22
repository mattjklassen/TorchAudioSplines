# ----- Brief Description -----
#
# Plot cubic spline f(t) with bcoeffs coming from file as arg1 on command line, and
# optional knot sequence from file as arg2, and optional inputs from file as arg3.
# Default knot sequence is 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1 and if inputs are
# given then they are also plotted as points in red. (Inputs are not needed for plot)
#
# ----- ----- ----- ----- -----

# ------- More Details --------
#
# Use default knot sequence 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1 unless second
# command line arg2 gives other knot sequence.  From knot sequence of length N+1,
# the dimension of cubic splines is N-3 = n which needs to match the number of bcoeffs.
# Example: For the cubic "splinusoid" we match sin(2Pix) on [0,1] at 9 points which are
# endpoints in a uniform partition into 8 subintervals.  Doing half of this problem
# first on [0,1/2] approximating sin(Pi*x) at 5 points works nicely with knot sequence:
# 0,0,0,0,1/4,1/2,3/4,1,1,1,1.  Then joining this with the negative of those B-spline
# coefficients on [1/2,1] gives the full period of sin(2Pix) approximation on [0,1].  
# The full knot sequence is then 0,0,0,0,1/8,2/8,3/8,4/8,4/8,4/8,4/8,5/8,6/8,7/8,1,1,1,1.
#
# ----- ----- ----- ----- -----

import torch
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from getBcoeffs import import_bcoeffs, export_bcoeffs
from getKnots import import_knots, export_knots

# The next two functions are imported to compute using the deBoor algorithm.  The first one computes
# the value of a single B-spline basis function for value t in [0,1].  This is necessary in order to
# set up the linear system to solve for bcoeffs (B-spline coefficients).  The second one computes a 
# value for B-spline function given all n=k+d bcoeffs.  

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 
from computeBsplineVal import computeSplineVal2 

print("Argument List:", str(sys.argv))

special_knots = 0

bcoeffs_file = sys.argv[1]
if len(sys.argv) > 2 :
    knots_file = sys.argv[2]
    special_knots = 1
if len(sys.argv) > 3 :
    inputs_file = sys.argv[3]

# file = "bcoeffs-sin2.txt"
bcoeffs = import_bcoeffs(bcoeffs_file)

# for this set of bcoeffs we need knot sequence: 0,0,0,0,1/8,2/8,3/8,4/8,4/8,4/8,4/8,5/8,6/8,7/8,1,1,1,1
# with N = 17.  Then dimension of cubic splines with this knot sequence is n = 14 = N-d.
# (Also dim n = d+1 + sum of multiplicities m_i at each breakpoint = 4 + 1+1+1+4+1+1+1 = 14.)
n = len(bcoeffs)  # n = 14
d = 3
# k is number of subintervals, which for splinusoid example is: [0,1/8,2/8,...,7/8,1] so k = 8
# for this example k is not simply n-d, which is the case for simple knots between interval endpoints
# such the default type knot sequence 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1.
# k = 8 can be computed from the breakpoint sequence in the splinusoid example by checking for
# number of unique knots between 0 and 1, below ...

# N is last index of knot sequence of length N+1
N = n + d   # N = 17
knotVals = torch.zeros(N+1)

# construct standard knot sequence: 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1 with k = n-d = n-3
k = n - d
incr = 1/k
for i in range(k-1) :
    knotVals[i+4] = knotVals[i+3] + incr
for i in range(4) :
    knotVals[N-i] = 1.0
print("standard knotVals:")
print(knotVals)
title = "cubic spline from bcoeffs file: " + bcoeffs_file

def make_special_knotVals() :
    # construct special knot sequence for splinusoid: 
    # 0,0,0,0,1/8,2/8,3/8,4/8,4/8,4/8,4/8,5/8,6/8,7/8,1,1,1,1
    for i in range(4) :
        j = i + 7
        knotVals[j] = 1/2
        j = i + 14
        knotVals[j] = 1
    
    incr = 1 / 8  # for this special knot sequence we need to already know k = 8
    for i in range(3) :
        j = i + 4
        knotVals[j] = knotVals[j-1] + incr
        j = i + 11
        knotVals[j] = knotVals[j-1] + incr
    
    print("special knotVals for splinusoid with n = 14: ")
    print(knotVals)
    # export_knots(file, knotVals)

if len(sys.argv) > 2 :
    knotVals = import_knots(knots_file)
    print("imported knotVals from file ", knots_file)
    print(knotVals)
    # now compute k as number of unique knots between 0 and 1, for splinusoid
    # otherwise standard knot sequence has simple def of k = n-d
    k = 0
    i = d + 1
    knot = 0.0
    while knot < 1 :
        knot = knotVals[i]
        if knot > knotVals[i-1] :
            k += 1
        i += 1
    title = "cubic spline matching sin(2Pi*x) on [0,1]"

# inputVals are endpoints of k subintervals
inputVals = torch.zeros(k+1)
incr = 1 / k
for i in range(k+1) :
    inputVals[i] = i * incr
print("inputVals:  ", inputVals)
print("k, d, n, N:  ", k, d, n, N)

outputVals = torch.zeros(k+1)
outputVals[k] = 0
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
# title = "cubic spline matching sin(2Pi*x) on [0,1]" - this is assigned earlier
# plt.title('cubic interpolating spline n=12, start and end at zero, 10 random values in [-1,1]')
plt.title(title)
plt.xlabel('time axis')
# plt.ylabel('audio sample axis')
plt.show()

