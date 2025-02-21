
# ----- Brief Description -----
# 
# Here we do glissandos between key cycles, morphing both cycle length and cycle shape
# Use quadratic version for exact fit of glissando to time interval.
# Generate waveform using bcoeffs of each key cycle with glissando and bcoeff interpolation between.
# bcoeffs.txt has list of bcoeffs files, as first command line parameter.
# Each cycle should use the same number n of bcoeffs for consistency.
# cycles.txt has starting and ending values [a,b] for each cycle which are float samples, 
# and has a length b - a in float samples, which converts to f0 in Hz.
# time = total length of waveform in seconds
# python multi-gliss2.py bcoeffs.txt cycles.txt time 
# 
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# first cycle is in middle of some chunk, say 1000 samples, so there is a lead in space of about
# 200-400 samples, which can be either nothing or some simple fade in, like one more cycle at half gain.
# It could be useful to have a funtion that inserts wav form between two key cycles by writing the first cycle
# and interpolated intermediate cycles but does not write the last one (even though it is the final target).
# This way we write sequentially and do not overlap.  The last cycle can then mimic the first by one more
# copy of that cycle at half gain.  We can modify insertWavTone2 below:
# insertWavTone2(waveform, 
#        start_time,  
#        f, 
#        time1, 
#        sample_rate, 
#        key_bcoeffs, 
#        key_knots, 
#        note_keys[i], 
#        key_gains, 
#        interp_method)
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

bcoeffs_file = sys.argv[1]   # text file of bcoeffs files, each with same number n
cycles_file = sys.argv[2]    # text file of cycles, one per line: a, b
time = float(sys.argv[3])    # in float samples

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

def genMultiGliss2(time, sample_rate, bcoeffs, cycles) :

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

    # write cycles to waveform buffer:
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


wav_data = genMultiGliss2(time, sample_rate, bcoeffs, cycles)
print("we have wav data")

# write wav_data to file:
size_out = int(sample_rate * time)
waveform = torch.empty(1, size_out)
for i in range(size_out) :
    waveform[0,i] = wav_data[i]

path1 = "../audio"
path = path1 + "/multi-gliss2" 
path += ".wav"
print("now writing wav file:")
print(path)

torchaudio.save(
    path, waveform, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)




