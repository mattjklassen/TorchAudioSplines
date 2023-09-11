# This program takes the following input on the command line:
# [1] wav file 
# [2] start sample number
# [3] end sample number
# [4] n (number of spline interpolation points = dimension of cubic spline vector space)

# output graph with matplot is the original audio graph as piecewise linear function
# overlayed with spline curve in green and interpolation points highlighted in red.

# Important note: the spline interpolation points are not uniform. We did this when
# forming splines to model cycles by choosing first a uniform sequence of k subintervals
# for the cubic spline and a knot sequence for the basis of B-splines for the vector 
# space V of C^2 cubic splines on the intervals [0,1/k,2/k,...,(k-1)/k,1].  We also chose
# the knot sequence 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1 since this gives a B-spline basis
# with the simple properties that for any f(t) given as a sum of the basis B-splines
# f(t) = SUM_{i=0}^{n-1} c_i B^3_i(t), where n=k+3 is the dimension of V, we have
# f(0) = c_0, f(1) = c_{n-1}, f'(0) = c_1, and f'(1) = c_{n-2}.  Further, the interpolation
# points (s_i,y_i) i=0,...,n-1 should be chosen on the interval [0,1] according to the
# Schoenberg-Whitney Theorem so that B^3_i(s_i) > 0, or equivalently such that 
# t_i < s_i < t_{i+4} for each i.  A uniform sequence s_i = i/(n-1) works for i=0,...,n-1
# but we chose to use the sequence of k+1 endpoints of subintervals and then to insert
# two more values to total n=k+3 at the midpoints of the first and last subintervals,
# at 1/(2k) and 1-1/(2k).  These values have the function of giving a slightly "tighter"
# fit of the spline at the endpoints.  Since we are fitting to audio data of one cycle
# at a time, this makes it more likely that there will be good agreement of derivatives 
# where cycle splines join at the endpoints.

import torch
import wave
import matplotlib.pyplot as plt
import numpy as np
import sys

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 

print("Argument List:", str(sys.argv))

audiofile = sys.argv[1]
start_sample = int(sys.argv[2])
end_sample = int(sys.argv[3])
n = int(sys.argv[4]) # dimension of C^2 cubic spline vector space V
# count = number of samples for graph, ends included
count = end_sample - start_sample + 1 

d = 3     # degree for cubic splines
k = n - d # number of subintervals
N = n + d # last index of knot sequence t_i, i=0,...,N
print("k, d, n, N:  ", k, d, n, N)

# arrays for building splines
knotVals = torch.zeros(N+1)
inputVals = torch.zeros(n)
outputVals = torch.zeros(n)

obj = wave.open(audiofile, "rb")
sr = obj.getframerate()
start_time = start_sample / sr
end_time = end_sample / sr
length_in_sec = count / sr
nsamp = obj.getnframes()
signal = obj.readframes(-1)

obj.close()

signal_len = nsamp / sr

print("sample rate:  ", sr)
print("length of audio file in samples:  ", nsamp)
print("length of audio file in seconds:  ", signal_len)
print("length of selection in samples:  ", count)
print("length of selection in seconds:  ", length_in_sec)

data_int = np.frombuffer(signal, dtype=np.int16)
data = data_int.astype('float32')
data /= 32768.0
times = np.linspace(0, signal_len, num=nsamp)

steps = count + 1
short_times = np.linspace(start_time, end_time, num=steps)
short_data = np.zeros(steps)
for i in range(0, count + 1) :
    short_data[i] = data[start_sample + i]

# We want to match audio data by computing input/output values using piecewise linear function
# through the audio samples on the time interval I=[start_time, end_time].  We do this to match
# the type of interpolation points we are choosing for the spline on the interval [0,1], by
# first setting up k+1=n-2 subinterval endpoints uniformly on the interval I, then inserting two
# more input values at the midpoints of outer subintervals, to get sequence interp_times.

temp_times = np.linspace(start_time, end_time, num=n-2)
temp_incr = (end_time - start_time) / k
temp_times[0] = start_time + temp_incr / 2
temp_times[n-3] = end_time - temp_incr / 2
# we will use this as the middle of the sequence of inputs s_i
interp_times = np.zeros(n)
for i in range(1, n-1) :
    interp_times[i] = temp_times[i-1]
interp_times[0] = start_time
interp_times[n-1] = end_time
# now interp_times play the role of inputs s_i on interval I.
# (we could also get interp_times just by scaling and shifting inputVals from[0,1])
print("interp times:  ", interp_times)

interp_data = np.zeros(n)
# end points are same as sample data
interp_data[0] = short_data[0]
interp_data[n - 1] = short_data[steps - 1]
# other points are computed on data graph as piecewise linear
for i in range(1, n - 1) :
    j = 0
    time0 = 0
    data0 = 0
    time1 = 0
    data1 = 0
    while short_times[j] < interp_times[i] :
        data0 = short_data[j]
        data1 = short_data[j+1]
        time0 = short_times[j]
        time1 = short_times[j+1]
        j += 1
    t = interp_times[i]
    denom = time1 - time0
    c0 = (t - time0) / denom
    c1 = (t - time1) / denom
    # (t,y) is on line between (time0,data0) and (time1,data1)
    y = c0 * data1 - c1 * data0
    interp_data[i] = y

print("interp data:  ", interp_data)
# continuing with spline setup, targets for spline from audio data:
targets = interp_data

# print("targets:  ", targets)

# subinterval size:
incr = 1 / k
# print("incr:  ", incr)

# Use inputs along the interval [0,1], first using the subinterval endpoints
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

# In previous version we assumed bcoeffs c[0]=0=c[n-1], but now we allow these to be nonzero.
# Since they still give the function value at the ends, we can simply set c[0]=data[0] and 
# c[n-1]=data[n-1], then need to solve for remaining c[i].  Also, the points will not all
# be uniformly spaced, since we use k+1 uniform on [0,1], then two more at 1/2k and 1-1/2k.

# Next, set up the linear system coefficient matrix A to solve for B-spline coeffs
# using B-spline evaluation function.

# Linear system rows i, columns j to solve for c[0] ... c[n-1] so
# the entry A[i,j] should be B^3_j(s_i) for input s_i. 

# the system, to solve for c_0,...,c_{n-1} 
# for function f(t) = sum of c_j*B^3_j(t) j=0,...,n-1
# looks like:
# row 0: [B^3_0(s_0) B^3_1(s_0) ... B^3_{n-1}(s_0)]
# row 1: [B^3_0(s_1) B^3_1(s_1) ... B^3_{n-1}(s_1)]
# where s_0=0, s_1=1/2k, s_2=1/k, s_3=2/k, ... , s_{n-3}=1-1/k, s_{n-2}=1-1/2k and s_{n-1}=1
# row n-1: [B^3_0(s_{n-1}) B^3_1(s_{n-1}) ... B^3_{n-1}(s_{n-1})]

A = torch.zeros(n, n)
# print("matrix A =  ", A)

for i in range(n) :
    for j in range(n) :
        A[i, j] = newBsplineVal(3, k, j, inputVals[i])
# print("matrix A =  ", A)

# print("type(targets):  ", type(targets))

b = torch.from_numpy(targets).float()
# print("torch.dtype(b):  ", torch.dtype(b))

# Now need to solve A*c=b for c (bcoeffs vector c) with b = targets
c = torch.linalg.solve(A, b)
print("bcoeffs vector c =  ", c)

# Next graph spline function (xvals, yvals) with the computed coefficients using 1000 points.

xvals = np.linspace(start=0.0, stop=1.0, num=1000)
# print(xvals)
# print("size of xvals:  ", xvals.size)
yvals = np.zeros(1000)
for i in range(1000) :
    t = xvals[i]
    yvals[i] = computeSplineVal(d, k, c, t)

# print(yvals)
# print("size of yvals:  ", yvals.size)

# rescale xvals for spline curve plot
time_interval = end_time - start_time
time_incr = 0.001 * time_interval
xvals = np.linspace(start_time, end_time, num=1000)

plt.figure(figsize=(15,8))
# plt.plot(times, data)
plt.plot(short_times, short_data, '0.8')
plt.plot(xvals, yvals, 'g')
plt.title(audiofile + " segment: samples " + str(start_sample) + " to " + str(end_sample) + " piecewise linear in grey, spline in green")
plt.ylabel("sample float values")
plt.xlabel("time in seconds")
# plt.xlim(0, signal_len)
plt.xlim(start_time, end_time)
plt.plot(interp_times, interp_data, 'ro')
plt.show()

# print("sample values:")
# for i in range(start_sample, end_sample) :
# print("sample number: ", i, "  sample value: ", data[i])


