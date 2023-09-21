# ----- Brief Description -----
#
# Plot cubic spline f(t) through n points (x,y) with x in [0,1], y in [-1,1]
# with f(0)=0=f(1) and other y-values randomly generated.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
#
# This program finds B-spline coefficients for spline to interpolate n target values.
# We use the default degree d=3 and k subintervals on [0,1], so the number of target values to
# interpolate is n = k + d.  We also use the standard knot sequence:
# 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1.  This makes it easy to start and end the spline
# with value 0 simply by choosing bcoeffs[0] = bcoeffs[n-1] = 0, and compute the other n-2 bcoeffs.
#
# ----- ----- ----- ----- -----

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# The next two functions are imported to compute using the deBoor algorithm.  The first one computes
# the value of a single B-spline basis function for value t in [0,1].  This is necessary in order to
# set up the linear system to solve for bcoeffs (B-spline coefficients).  The second one computes a 
# value for B-spline function given all n=k+d bcoeffs.  

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 

# change n here to increase the number of random interpolation points.
n = 15

# leave d=3 for cubic splines
d = 3
# k is number of subintervals
k = n - d 
# N is last index of knot sequence of length N+1
N = n + d
print("k, d, n, N:  ", k, d, n, N)

knotVals = torch.zeros(N+1)
inputVals = torch.zeros(n)
outputVals = torch.zeros(n)

# targets should be from audio sample, but are set to random for now (first and last = 0)
targets = np.random.uniform(low=-1.0, high=1.0, size=n-2)
print("targets:  ", targets)

# we will plot points (inputVals, outputVals) with spline plot
for i in range(1,n-1) :
    outputVals[i] = targets[i-1]

# subinterval size:
incr = 1 / k
print("incr:  ", incr)

# We have n random outputs, zero at ends and n-2 random values between -1 and 1.  These will
# correspond to inputs along the interval [0,1], first using the subinterval endpoints
# 0,1/k,2/k,...,(k-1)/k,1 (k+1 of these) and then two more values: 1/2k and 1-1/2k.
# The idea with using these last two values is that they give a little more information near
# the endpoints which can affect the slope at the ends when interpolating continuous data.
inputVals[0] = 0.0
inputVals[1] = 0.5 * incr
inputVals[2] = incr  # after this one, just add incr to fill in between
inputVals[n-2] = 1 - 0.5 * incr
inputVals[n-1] = 1.0

for i in range(3,n-2) :
    inputVals[i] = inputVals[i-1] + incr
print("inputs:  ", inputVals)

for i in range(N+1) :
    knotVals[i] = (i-d) * incr
    if (i < d) :
        knotVals[i] = 0
    if (i > N-d-1) :
        knotVals[i] = 1
print("knots:  ", knotVals)

# Assume bcoeffs c[i], with c[0]=0=c[n-1], so need to solve for remaining c[i].
# Next, set up the linear system coefficient matrix A to solve for B-spline coeffs
# using B-spline evaluation function.

# Linear system rows i, columns j to solve for c[1] ... c[n-2]
# in system Ax=b these are indexed 0 ... n-3
# the entry A[i,j] should be B^3_j(s_i) for input s_i, but B-splines
# are shifted forward by one index, so B^3_{j+1}(s_{i+1})

# the "whole" system, to solve for c_0,...,c_{n-1} 
# for function f(t) = sum of c_j*B^3_j(t) j=0,...,n-1
# would look like:
# row 0: [B^3_0(s_0) B^3_1(s_0) ... B^3_{n-1}(s_0)]
# row 1: [B^3_0(s_1) B^3_1(s_1) ... B^3_{n-1}(s_1)]
# where s_0=0 and s_{n-1}=1
# row n-1: [B^3_0(s_{n-1}) B^3_1(s_{n-1}) ... B^3_{n-1}(s_{n-1})]
# but from the choice of knot sequence we have f(0)=f(1)=0 with c_0=c_{n-1}=0
# so we eliminate B^3_0 and B^3_{n-1} from the system and also c_0 and c_{n-1}
# then when evaluating B-splines we use deBoor algorithm with the chosen knot sequence
# and d=3, and subscripts j=0,...,n-1 in order to extract the above values B^3_j(s_i)
# for j=1,...,n-2 only.  The smaller system n-2 by n-2 should eliminate both the first
# and last rows and columns. 

A = torch.zeros(n-2, n-2)
# print("matrix A =  ", A)

for i in range(n-2) :
    for j in range(n-2) :
        A[i, j] = newBsplineVal(3, k, j+1, inputVals[i+1])
print("matrix A =  ", A)

# print("type(targets):  ", type(targets))

b = torch.from_numpy(targets).float()
# print("torch.dtype(b):  ", torch.dtype(b))

# Now need to solve A*x=b for x with b = targets
v = torch.linalg.solve(A, b)
print("nonzero bcoeffs solution v =  ", v)

# put zeros at ends of bcoeffs vector c, which gives dimension n
c = np.zeros(n)
for i in range(n) :
    if (i>0) and (i<n-1) :
    	c[i] = v[i-1]
print("bcoeffs vector c:  ", c)

# Next graph spline function with the computed coefficients using 1000 points.

xvals = np.arange(start=0.0, stop=1.001, step=.001)
print(xvals)
print("size of xvals:  ", xvals.size)
yvals = np.zeros(1001)
for i in range(1001) :
    t = xvals[i]
    yvals[i] = computeSplineVal(d, k, c, t)

print(yvals)
print("size of yvals:  ", yvals.size)
plt.figure(figsize=(15,8))
plt.plot(xvals, yvals)
plt.plot(inputVals, outputVals, 'ro')
title = "cubic interpolating spline n="
title += str(n)
title += ", start and end at zero, "
title += str(n-2)
title += " random values in [-1,1]"
# plt.title('cubic interpolating spline n=12, start and end at zero, 10 random values in [-1,1]')
plt.title(title)
plt.xlabel('time axis')
# plt.ylabel('audio sample axis')
plt.show()

