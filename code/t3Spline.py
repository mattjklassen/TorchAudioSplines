# ----- Brief Description -----
#
# Plot cubic spline f(t) through n points (x,y) with x in [0,1], y in [-1,1]
# with f(0)=0=f(1), and f'(0)=Pi, f'(1)=-Pi, f(1/2)=1, f(1/4)=2^(-1/2)=f(3/4).
# These seven conditions match the function y = sin(pi*x) on [-1,1].
# So n=7, k=4, d=3.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
#
# The above conditions give a 7 by 7 system on the B-spline coefficients.
# The first and last rows and cols can be eliminated since we know those two
# correspond to first and last coefficients being zero.  The first and last
# rows and columns of the remaining system need to be included, so the final
# system to solve is 5 by 5.  The coefficient c_1 must be Pi/12 since the 
# only nonzero derivative is B^3_1'(0)=12 and the target value is Pi. 
# Similarly c_5=Pi/12.  The other entries in first and last rows need to be
# entered as zero.  The system we have assumes rows come from input/output
# pairs, but these rows are derivative based.  So we can overwrite them.
# There is one other nonzero value to enter into the 5 by 5 system in two 
# places, which is B^3_1(1/4)=B^3_5(3/4) in the first column second row, and 
# the last column fourth row.  We can let the system compute those.
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
n = 7

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
targets[0] = 0.7071067812  # 2^(-1/2) = sin(Pi/4)
targets[1] = 1.0           #          = sin(Pi/2)
targets[2] = 0.7071067812  # 2^(-1/2) = sin(3*Pi/4)
print("targets:  ", targets)

# we will plot points (inputVals, outputVals) with spline plot,
# first and last outputs are already zero, so set the n-4 interior values to targets
for i in range(1,n-3) :           # i goes 1 to n-4
    outputVals[i] = targets[i-1]

# subinterval size:
incr = 1 / k
print("incr:  ", incr)

# We have n outputs, zero at ends and zero derivative at ends, and n-4 target values 
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

# Assume bcoeffs are c[i], with c[0]=0, c[1]=Pi/12, c[n-2]=-Pi/12, c[n-1]=0, 
# so need to solve for remaining c[i] for i=2,...,n-3, or bcoeffs c[2],c[3],c[4].
# Next, set up the linear system coefficient matrix A to solve for B-spline coeffs
# using B-spline evaluation function.  From the remarks above, we choose to use
# the 5 by 5 system with modified first and last rows.

# linear system: rows i, columns j to solve for c[1] ... c[n-2]
# in system Ax=b these are indexed 0 ... n-3
# the entry A[i,j] should be B^3_j(s_i) for input s_i, but B-splines
# are shifted forward by one index, so B^3_{j+1}(s_{i+1})

# the "whole" system, to solve for c_0,...,c_{n-1} 
# for function f(t) = sum of c_j*B^3_j(t) j=0,...,n-1
# would look like: (without derivative rows)
# row 0: [B^3_0(s_0) B^3_1(s_0) ... B^3_{n-1}(s_0)]
# row 1: [B^3_0(s_1) B^3_1(s_1) ... B^3_{n-1}(s_1)]
# where s_0=0 and s_{n-1}=1
# row n-1: [B^3_0(s_{n-1}) B^3_1(s_{n-1}) ... B^3_{n-1}(s_{n-1})]

# but from the choice of knot sequence we have: f(0)=f(1)=0, f'(0)=Pi, and f'(1)=-Pi with 
# c_0=c_{n-1}=0 and c_1=Pi/12, c_{n-2}=-Pi/12, so we eliminate B^3_0, B^3_{n-1}, but we allow
# B^3_1 and B^3_{n-2}.  

A = torch.zeros(n-2, n-2)
# print("matrix A =  ", A)

for i in range(n-2) :
    for j in range(n-2) :
        A[i, j] = newBsplineVal(3, k, j+1, inputVals[i])
        # set first and last rows with correct zeros: 
        if i == 0 and j > 0 :
            A[i, j] = 0
        if i == n-3 and j < n-3 :
            A[i, j] = 0
A[0,0] = 1
A[n-3,n-3] = 1
print("matrix A =  ")
print(A)

# print("type(targets):  ", type(targets))

b = torch.zeros(n-2)
b[0] = 3.1415926535 / 12
b[n-3] = 3.1415926535 / 12
for i in range(n-4) :
    b[i+1] = targets[i] 
print("b:  ", b)

# Now need to solve A*x=b for x with b = targets
v = torch.linalg.solve(A, b)

print("nonzero bcoeffs solution v =  ", v)
print("shape of v: ", v.shape)
print("len of v: ", len(v))

# put zeros at ends of bcoeffs vector c, which gives dimension n
c = np.zeros(n)
for i in range(n) :
    if (i>0) and (i<n-1) :
    	c[i] = v[i-1]

# Note: Since f(0)=0 by setting c_0=0, we get f'(0)=c_1*B^3_1'(0)=c_1*12, similar for f'(1)=c_5*(-12).

print("bcoeffs vector c:  ", c)
print("exporting to bcoeffs12.txt")

bcoeffs = []
for i in range(len(c)) :
    bcoeffs.append(float(c[i]))

file = "bcoeffs-sin.txt"
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
title = "cubic spline n="
title += str(n)
title += ", matching sin(Pi*x) on [0,1] at 5 points and derivatives at ends"
plt.title(title)
plt.xlabel('time axis')
plt.show()

