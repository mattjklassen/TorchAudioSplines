
# ----- Brief Description -----
# 
# Generate waveform using bcoeffs of one cycle with glissando.
# Bcoeffs file is first command line parameter.
# f01 = starting frequency, f02 = ending frequency
# time = length of glissando in seconds
# python genGlissando.py bcoeffs.txt f01 f02 time 
# 
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Here we need to solve the problem of interpolation of cycle length stated as follows: 
# Given Time L in seconds, and frequencies f01 > f02 with cycle lengths c1 < c2, partition 
# the interval of length L into k+2 cycles or subintervals, of lengths l_i, i = 0 to k+1,
# with l_0=c1 and l_{k+1}=c2, such that l_0 < l_1 < ... < l_{k+1}.  Assuming a solution exists 
# (there are cases where this does not hold, easily constructed for example where L-c1-c2 < c1, etc.)
# we could also try to find one where c_i = c_{i-1}+a for some constant a.  But this requires
# that L/k be equal to the average of c1 and c2.  So some compromise is required.
# If it is OK to adjust L to L' then we can simply use k to be the smallest positive integer
# such that L' = k*(c1+c2)/2 < L, adjusting downward so L' = L-1.  This method is best
# if we want to specify the time, in which case we write fewer cycles and do not fill entirely.
#
# ----- ----- ----- ----- -----

import torch
import torchaudio
import numpy as np
import math
import sys

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 
from genCycle import genCycle, insertCycle, insertCycle2, insertCycle3
from getBcoeffs import getBcoeffs, import_bcoeffs, export_bcoeffs
from genWavTone import reset
import pathlib


bcoeffs_file = sys.argv[1]
bcoeffs = import_bcoeffs(bcoeffs_file)

print("imported bcoeffs:")
print(bcoeffs)

n = bcoeffs.size(dim=0)
f01 = float(sys.argv[2])
f02 = float(sys.argv[3])
time = float(sys.argv[4])

sample_rate = float(44100.0)

print("Argument List:", str(sys.argv))
args = len(sys.argv)
print("There are ", args, " args")

if args > 5 :
    method = sys.argv[5]

print("frequences:")
print("start f01: ", f01)
print("end f02: ", f02)

path1 = "poly_tones/gliss" 

pathlib.Path(path1).mkdir(parents=True, exist_ok=True)


# generate basic waveform for tone of length time in seconds:

def genGliss(f01, f02, time, sample_rate, bcoeffs) :

    c1 = float(1.0 / f01) # starting cycle length in seconds
    c2 = float(1.0 / f02) # ending cycle length in seconds
    c1_samples = sample_rate * c1
    c2_samples = sample_rate * c2
    avg = (c1 + c2) / 2
    avg_samples = avg * sample_rate
    k = int(time / avg) - 2      # k = number of intermediate cycles
    k2 = (time / avg) - 2      # k2 = number of intermediate cycles float
    kf = k2 - k  # fractional part of k2
    size = (time - c1 - c2) / k  # this size divides up intermediate interval into k equal pieces
    size_samples = size * sample_rate
    print("k = int(time / avg) - 2 = ", k)
    print("k2 = (time / avg) - 2 = ", k2)
    print("kf = fractional part of k = ", kf)
    print("c1 = ", c1)
    print("c2 = ", c2)
    print("c1_samples = ", c1_samples)
    print("c2_samples = ", c2_samples)
    print("avg = ", avg)
    print("avg_samples = ", avg_samples)
    print("size = ", size)
    print("size_samples = ", size_samples)

    num_cycles = k + 2  # 2 for the ends, c1 and c2, and k in the middle
    waveform = torch.zeros(int(sample_rate * time) + 1) 
    incr = (c2-c1)/(k+1)
    delta = sample_rate * incr
    print("delta: ", delta) 

    # compute the sum of exp growth rate version of cycles using same k:
    exp = float(1 / (k+1))
    if c1 > c2 :
        exp = - exp
    # below we are using base 2 since R = f02/f01 = 2 = c1/c2 
    fac = pow(2, exp)
    expsum = c1_samples
    explen = c1_samples
    for i in range(1, k+2) :  # goes 1 to k+1
        explen *= fac
        expsum += explen

    print("expsum evaluated as sum with k = int(time/avg)-2 :\n")
    print("expsum:  ", expsum)
    print("last cycle length:")
    print("explen:  ", explen)
    print("\n")

    # for comparison compute sum with formula 
    w = fac
    expsum = sample_rate * c1 * (1 - w / 2) / (1 - w) 
    print("expsum (same k) evaluated with closed formula for sum of finite geometric series:")
    print("expsum:  ", expsum)
    print("\n")

    # now find newk so that expsum is < time but as close as possible
    newk = k
    while expsum < time * sample_rate :
        newk += 1
        exp = float(1 / (newk+1))
        if c1 > c2 :
            exp = - exp
        fac = pow(2, exp)
        w = fac
        expsum = sample_rate * c1 * (1 - w / 2) / (1 - w) 

    newk -= 1
    exp = float(1 / (newk+1))
    if c1 > c2 :
        exp = - exp
    fac = pow(2, exp)
    w = fac
    expsum = sample_rate * c1 * (1 - w / 2) / (1 - w) 

    print("expsum (with newk) as close as possible to time:")
    print("newk:  ", newk)
    print("expsum:  ", expsum)
    print("current cycle length: ", c2 * sample_rate)
    print("previous cycle length: ", (c2 / w) * sample_rate)
    print("error = ", time * sample_rate - expsum)
    print("\n")

    L = time
    print("compute exp version of k' or kp")
    kp = -1 - np.log(2) / np.log((L - c1)/(L-c2))
    k = int(kp)
    print("k' is:  ", kp)

    print("\n")

    print("all cycle lengths in samples and fundamental frequencies in Hz:")
    print("cycle 0 = ", c1_samples, "  f0: ", 1/c1)
    cj = c1
    for j in range(k) :
        cj = cj * w
        print("cycle ", j+1, " = ", cj * sample_rate, "  f0: ", 1/cj)

    print("cycle ", k+1, " = ", c2_samples, "  f0: ", 1/c2)


    # write cycles to waveform buffer:
    a = 0.0
    b = 0.0
    cycle_length = c1_samples
    # write cycles 
    for i in range(num_cycles) :  # numcycles = k+2
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        insertCycle(waveform, cycle, bcoeffs)
        # linear version with delta:
        cycle_length += delta
        # exp version with w:
        # cycle_length *= w

    print("last b: ", b)
    print("last cycle length:", cycle_length)
    print("last sample: ", int(sample_rate * time))
    print("error kf * avg = ", kf * avg_samples)

    return waveform


wav_data = genGliss(f01, f02, time, sample_rate, bcoeffs)
print("we have wav data")

# write wav_data to file:
size_out = int(sample_rate * time)
waveform = torch.empty(1, size_out)
for i in range(size_out) :
    waveform[0,i] = wav_data[i]

path1 = "../audio"
path = path1 + "/glissando" 
path += ".wav"
print("now writing wav file:")
print(path)

torchaudio.save(
    path, waveform, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)


# Linear growth rate of cycles is the most basic way to get the partition of time approximately
# into increasing intervals with constant difference delta between cycles.  It does not hit
# the total time exactly, however.  The simplest approach is to suppose we have two frequency
# values f01 and f02 with cycle lengths c01 = 1/f01 and c02 = 1/f02, and that these will make
# up the first and last cycle length in a time interval of length L.  We will want to fit k
# cycles in between these two such they the lengths form a monotone (increasing or decreasing)
# sequence from c01 to c02, and such that the sum of cycle lengths is L.  One starting point
# is suppose all intermediate cycles have the same length equal to the average of c01 and c02
# say avg = (c01+c02)/2.  Then if L' = L-c01-c02, we have L'/avg = k' is the (non-integer) number 
# of cycles to use. But we want an integer, so we can let k = floor(k').  

# Suppose the error = e, and we distribute e amongst the k intermediate cycles. This can be 
# done in various ways, either constant amounts or some other distribution.  We want to keep
# the starting and ending cycles the same, so maybe it is too messy. We also don't want to
# disturb the property of monotone sequence of lengths.  In the example f01=220, f02=440,
# and sample rate 44100, c01=200.45, c02=100.23, avg=150.34, time=L=2, we have k'=584.66, k=584.
# So the error is e = 0.66 * avg = 100.23 (same as c02 in this case).  If we distribute this
# error evenly amongst 584 intermediate cycles, we have only an increase of 100.23/584 = 
# 0.17 samples per cycle.  This is approximately equal to the delta value by which cycles
# are changing, so we would be doubling that.  

# Next we should do a version with quadratic interpolation.  We can use the Bernstein basis:
# f(t) = alpha * (1-t)^2 + delta * 2*(1-t)*t + beta * t^2, with f(0)=alpha, f(1)=beta, and 
# f(1/2)=(1/4)(alpha+beta) + (1/2)delta.  So if delta = (1/2)(alpha+beta) then f is linear.
# Assume alpha = c1, beta = c2, and delta is small and controls the offset which can then
# function as a correction term to give the exact interpolation for the chosen time.
# So we could change to:
# f(t) = alpha * (1-t)^2 + ((1/2)(alpha+beta) + delta)*2*(1-t)*t + beta * t^2
# so that delta = 0 corresponds to linear, delta > 0 gives a quadratic above linear.
# This would allow for error compensation as well as approaching a more log type curve.

# We can also do cubic interpolation with Bernstein basis:
# f(t) = alpha * (1-t)^3 + gamma * 3*(1-t)^2*t + delta * 3*(1-t)*t^2 + beta * t^3, 
# with f(0)=alpha, f(1)=beta, and f(1/2) = (1/8)*(alpha + beta) + (3/8)*(gamma + delta).
# We can then specify f'(0) = f'(1) = 0 and f(1/2) = (1/2)*(alpha + beta).
# This version will be symmetric about the point (1/2, (1/2)*(alpha + beta))

# We should also consider an interpolation of cycle length which gives frequency interpolation
# which is logarithmic, for psychoacoustic effect. For example, suppose we have determined k, 
# and we are going from c1 to c2 so that we have one octave increase, with c2 = (1/2)*c1. Then
# we can attempt to increase cycle lengths to map more closely to cent values.  This would mean
# that each step should have cent value increase given by 1200/(k+1) or frequency ratio increase
# 2^(1/(k+1)). Two consecutive cycles of lengths r > s can represent frequency ratio (1/s)/(1/r)
# = r/s which should be = 2^(1/(k+1)).  So the next cycle length s should be r * 2^(-1/(k+1)).  
# Then the last step would achieve c1 * 2^(-1/(k+1))^(k+1) = c1 * (1/2) = c2. With this scheme
# we get to the tritone half way through the sequence of cycles.  

# But what do we get for the sum of cycle lengths? 

# The sum is now c1 * SUM_{i=0}^{k+1} 2^{-1/(k+1)}^i = c1 * (1-2^{-1/(k+1)}^(k+2)) / (1-2^{-1/(k+1)})
# = c1 * (1 - (1/2)*w) / (1 - w) where w = 2^{-1/(k+1)}.  Solving for w gives:
# w = (L-c1)/(L-(1/2)c1) = R < 1.  So 2^{-1/(k+1)} = R, and k+2 = 1 - ln(2)/ln(R). 
# So we can use k'+2 = 1 - ln(2)/ln(R) and k+2 = floor(1 - ln(2)/ln(R)), or k = floor(k').

# When we change from doing an octave glissando to some other interval between f01 and f02 we can
# replace 2 by the frequency ratio f02/f01, or we can use the cent value for this frequency ratio
# x = (1200/ln(2))*ln(f02/f01) which expresses f02/f01 = 2^{x/1200}.  The gliss can then be
# achieved to go from f01 to f02 by increasing or decreasing cycle lengths c01 to c02 by using
# the powers (for x>0) 2^{(x/1200)*(1/(k+1))*j}, for j = 1,...,k+1.

# Important note:  After running a few examples and comparing outputs, linear vs exponential growth
# of cycle lengths, it seems that there is very little noticeable difference.  This likely is
# because we are already getting some spread in frequency values when cycle length is changing
# linearly since f0 is 1/c0, and higher frequency has greater absolute value of slope for f(x)=1/x.




