# ----- Brief Description -----
#
# Plot cubic spline f(t) through n points (x,y) with x in [0,1], y in [-1,1]
# with f(0)=0=f(1), and f'(0)=0=f'(1) and other y-values randomly generated.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
#
# This program finds B-spline coefficients for spline to interpolate n-2 target values
# and two derivative values = 0 at ends.
# We use the default degree d=3 and k subintervals on [0,1], so the number of target values to
# interpolate is n = k + d.  We also use the standard knot sequence:
# 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1.  This makes it easy to start and end the spline
# with value 0 simply by choosing bcoeffs[0] = bcoeffs[n-1] = 0, 
# and with derivative 0 at ends by choosing bcoeffs[1] = bcoeffs[n-2] = 0 
# and then compute the other n-4 bcoeffs.  For example, a spline with 7 target values
# at the endpoints of 6 subintervals on [0,1] has k=6, d=3, n=9, and can be solved with
# 9 x 9 linear system for the B-spline coefficients, which is reduced to 5 x 5 with the
# above zero conditions.  The 5 remaining conditions are simply the values at the interior points.
# The resulting spline must coincide with the simple solution consisting of Hermite cubic
# polynomial p(x) on each subinterval given by endpoint conditions of two points and derivatives = 0.
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

# change n here to increase the number of random interpolation points.
n = 20

# leave d=3 for cubic splines
d = 3
# k is number of subintervals
k = n - d 
# N is last index of knot sequence of length N+1
N = n + d
print("k, d, n, N:  ", k, d, n, N)

knotVals = torch.zeros(N+1)
inputVals = torch.zeros(n-2)   # 0, 1/(n-3),2/(n-3),...,1=(n-3)/(n-3)
outputVals = torch.zeros(n-2)  # y_0=0,y_1,...,y_(n-4),y_(n-3)=0

# targets should be from audio sample, but are set to random for now (first and last = 0)
targets = np.random.uniform(low=-0.9, high=0.9, size=n-4)
print("targets:  ", targets)

# we will plot points (inputVals, outputVals) with spline plot,
# first and last outputs are already zero, so set the n-4 interior values to targets
for i in range(1,n-3) :           # i goes 1 to n-4
    outputVals[i] = targets[i-1]

# subinterval size:
incr = 1 / k
print("incr:  ", incr)

# We have n outputs, zero at ends and zero derivative at ends, and n-4 random values 
# between -1 and 1.  These will correspond to inputs along the interval [0,1], using 
# the subinterval endpoints 0,1/k,2/k,...,(k-1)/k,1 (k+1=n-2 of these) and then two more 
# conditions given as derivative zero at ends.

for i in range(1,n-2) :
    inputVals[i] = inputVals[i-1] + incr
print("inputs:  ", inputVals)

for i in range(N+1) :
    knotVals[i] = (i-d) * incr
    if (i < d) :
        knotVals[i] = 0
    if (i > N-d-1) :
        knotVals[i] = 1
print("knots:  ", knotVals)

# Assume bcoeffs are c[i], with c[0]=c[1]=0=c[n-2]=c[n-1], so need to solve for remaining 
# c[i] for i=2,...,n-3, or bcoeffs c[2],...,c[n-3].
# Next, set up the linear system coefficient matrix A to solve for B-spline coeffs
# using B-spline evaluation function.

# Linear system rows i, columns j to solve for c[2] ... c[n-3]
# in system Ax=b these are indexed 0 ... n-5
# the entry A[i,j] should be B^3_j(s_i) for input s_i, but B-splines
# are shifted forward by two indices, so B^3_{j+2}(s_{i+2})

# the "whole" system, to solve for c_0,...,c_{n-1} 
# for function f(t) = sum of c_j*B^3_j(t) j=0,...,n-1
# would look like:
# row 0: [B^3_0(s_0) B^3_1(s_0) ... B^3_{n-1}(s_0)]
# row 1: [B^3_0(s_1) B^3_1(s_1) ... B^3_{n-1}(s_1)]
# where s_0=0 and s_{n-1}=1
# row n-1: [B^3_0(s_{n-1}) B^3_1(s_{n-1}) ... B^3_{n-1}(s_{n-1})]

# but from the choice of knot sequence we have: f(0)=f(1)=0=f'(0)=f'(1) with 
# c_0=c_1=c_{n-2}=c_{n-1}=0 so we eliminate B^3_0, B^3_1, B^3_{n-2} and B^3_{n-1} 
# from the system and also c_0, c_1, c_{n-2} and c_{n-1}
# then when evaluating B-splines we use deBoor algorithm with the chosen knot sequence
# and d=3, and subscripts j=0,...,n-1 in order to extract the above values B^3_j(s_i)
# for j=2,...,n-3 only.  The smaller system n-4 by n-4 should eliminate the first and
# last two rows and columns. 

A = torch.zeros(n-4, n-4)
# print("matrix A =  ", A)

for i in range(n-4) :
    for j in range(n-4) :
        A[i, j] = newBsplineVal(3, k, j+2, inputVals[i+1])
print("matrix A =  ", A)

# print("type(targets):  ", type(targets))

b = torch.from_numpy(targets).float()
# print("torch.dtype(b):  ", torch.dtype(b))

# Now need to solve A*x=b for x with b = targets
v = torch.linalg.solve(A, b)

print("nonzero bcoeffs solution v =  ", v)
print("shape of v: ", v.shape)
print("len of v: ", len(v))

# put zeros at ends of bcoeffs vector c, which gives dimension n
c = np.zeros(n)
for i in range(n) :
    if (i>1) and (i<n-2) :
    	c[i] = v[i-2]
print("bcoeffs vector c:  ", c)
print("exporting to bcoeffs12.txt")

bcoeffs = []
for i in range(len(c)) :
    bcoeffs.append(float(c[i]))

file = "bcoeffs12.txt"
export_bcoeffs(file, bcoeffs)

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

