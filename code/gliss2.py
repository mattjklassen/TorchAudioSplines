
# ----- Brief Description -----
# 
# Quadratic version for exact fit of glissando to time.
# Generate waveform using bcoeffs of one cycle with glissando.
# Bcoeffs file is first command line parameter.
# f01 = starting frequency, f02 = ending frequency
# time = length of glissando in seconds
# python gliss2.py bcoeffs.txt f01 f02 time 
# 
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Here we need to solve the problem of interpolation of cycle length stated as follows: 
# Given Time L in seconds, and frequencies f01 > f02 (or <) with cycle lengths c_0 < c_1 (or >), 
# partition the interval of length L into k cycles or subintervals, of lengths L_i, i = 0 to k+1,
# with L_0=c_0 and L_{k+1}=c_1, such that L_i, i=0,...,k+1 is a  monotone sequence.  Assuming a 
# solution exists (there # cases where this does not hold, easily constructed for example where 
# L-c1-c2 < c1, etc) we use q(t) = c_0 (1-t^2) + (a+delta) 2(1-t)t + c_1 t^2, where a = (c_0+c_1)/2
# and delta will be used to set the time to L.  The integer k is equal to floor(L'/a), where L'
# is L-c_0-c_1.  We also set k'=L'/a, and x=k'-k. The values L_i = q(i/(k+1)) for i=0,...,k+1.
# To get the sum of the L_i to equal L we use delta = (L/(k+2)-a)/(1-(2k+3)/(3k+3)).
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
f01 = float(sys.argv[2])
f02 = float(sys.argv[3])
time = float(sys.argv[4])

bcoeffs = import_bcoeffs(bcoeffs_file)
print("imported bcoeffs:")
print(bcoeffs)
n = bcoeffs.size(dim=0)

sample_rate = float(44100.0)

print("Argument List:", str(sys.argv))
args = len(sys.argv)
print("There are ", args, " args")

if args > 5 :
    method = sys.argv[5]

print("frequences:")
print("start f01: ", f01)
print("end f02: ", f02)

# path1 = "poly_tones/gliss" 
# pathlib.Path(path1).mkdir(parents=True, exist_ok=True)

# generate waveform for tone of length time in seconds:

def genGliss2(f01, f02, time, sample_rate, bcoeffs) :

    L = time * sample_rate # total time interval in samples
    C0 = float(1.0 / f01) # starting cycle length in seconds
    C1 = float(1.0 / f02) # ending cycle length in seconds
    c0 = sample_rate * C0 # starting cycle length in samples
    c1 = sample_rate * C1 # ending cycle length in samples
    Lp = L - c0 - c1      # = L' intermediate interval in samples
    avg = (c0 + c1) / 2   # = average of first and last cycles in samples
    kp = (Lp / avg)       # kp = number of intermediate cycles float = k'
    k = int(kp)         # k = number of intermediate cycles
    x = kp - k          # fractional part of kp
    print("k = ", k)
    print("kp = ", kp)
    print("x = ", x)
    print("c0 = ", c0)
    print("c1 = ", c1)
    print("avg = ", avg)

    num_cycles = k + 2  # 2 for the ends, c0 and c1, and k in the middle
    numer = L / float(k+2) - avg
    denom = 1 - float(2*k+3) / float(3*k+3)  # this is approx = 1/3
    delta = numer / denom
    print("delta: ", delta) 
    waveform = torch.zeros(int(L) + 1)

    print("\n")

    # write all cycles (including key cycles) to waveform buffer:
    a = 0.0
    b = 0.0
    previous = 0
    for i in range(num_cycles) :  # numcycles = k+2
        # write cycle i
        a = b
        # quadratic version with delta:
        t = i / float(k+1)
        q1 = c0 * (1-t) * (1-t)
        q2 = 2 * (avg + delta) * (1-t) * t
        q3 = c1 * t * t
        cycle_length = q1 + q2 + q3  # cycle length in samples
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        insertCycle(waveform, cycle, bcoeffs)
        # if i < 40 :
#         print("t value: ", t)
#         print("computed cycle length for i = ", i)
#         print("cycle_length: ", cycle_length)
#         print("b value: ", b)
#         print("growth:  ", cycle_length - previous)
#         print("\n")
#         previous = cycle_length
#       exit(0)

    print("last b: ", b)
    print("last cycle length:", cycle_length)
    print("last sample: ", int(sample_rate * time))

    return waveform


wav_data = genGliss2(f01, f02, time, sample_rate, bcoeffs)
print("we have wav data")

# write wav_data to file:
size_out = int(sample_rate * time)
waveform = torch.empty(1, size_out)
for i in range(size_out) :
    waveform[0,i] = wav_data[i]

path1 = "../audio"
path = path1 + "/gliss2" 
path += ".wav"
print("now writing wav file:")
print(path)

torchaudio.save(
    path, waveform, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)




