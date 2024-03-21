
# ----- Brief Description -----
# 
# Continuing with testing a dulcimer tone using 32 key cycles, now we break those up into
# 17 subsequences of 16 consecutive key cycles, and construct a waveform in each case.
# 
# The key cycles are chosen from a 3 second long sample called dulcimerA3-f.wav with
# fundamental frequency f_0 = 220 Hz, so approximately 3*220 = 660 cycles, from which we chose 32.
# The bcoeffs files are given below.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----


import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from argMaxSpec import plotSpecArgMax, getArgMax
from cycleSpline import plotCycleSpline
from getCycles import getCycles
from getBcoeffs import import_bcoeffs, export_bcoeffs
from genWavTone import genWavTone


# test the function genWavTone.py with waveform data and write outputs to wav and pdf
# genWavTone(f0, time, sample_rate, key_bcoeffs, keys, gains, interp_method) 

# main part of script

f0 = 220.0
# f0 *= 1.059463
time = 3.0
# time = 0.25
sample_rate = 44100.0
# n = bcoeffs.size(dim=0)
n = 40
keys = torch.tensor([0,2,4,6,10,14,18,25,35,50,70,90,110,140,170,210])
print("keys: ", keys)
num_keys = keys.size(dim=0)
print("num_keys:  ", num_keys)

# this key sequence works for one second at frequency 220 Hz
# for shorter or longer time we can scale this as:
temp = time * keys
print("temp keys: ", temp)
for i in range(num_keys) :
    keys[i] = int(temp[i])
print("keys: ", keys)

gains = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
num_gains = len(gains)
print("num_gains:  ", num_gains)

for j in range(16) :
    if j > 6 :
        scale = 1.0 - float(j - 6) / 16
        gains[j] *= scale

print("scaled gains:")
print(gains)

num_gains = len(gains)
print("num_gains: ", num_gains)

interp_method = 1

# key cycles chosen from 16 segments: 0 through 10, and 20,30,40,50,60. 
# seg num     cycles
# 0           0,8,14,32
# 1           2,8,15,
# 2           0,7,17,
# 3           4,8,13
# 4           1,7,17
# 5           4,8,
# 6           3,9,
# 7           1,4,
# 8           2,5
# 9           0,4
# 10          3
# 20          2,
# 30          3,
# 40          10
# 50          9
# 60          0

# each label refers to segment and cycle numbers for bcoeffs file names for key cycles
labels = [[0,0],[0,8],[0,14],[0,32],[1,2],[1,8],[1,15],[2,0],[2,7],[2,17],[3,4],[3,8],[3,13],[4,1],[4,7],[4,17],[5,4],[5,8],[6,3],[6,9],[7,1],[7,4],[8,2],[8,5],[9,0],[9,4],[10,3],[20,2],[30,3],[40,10],[50,9],[60,0]]
# print("labels:  ", labels)
num_labels = len(labels)
# print("num_labels: ", num_labels)

print("entering loop on j to assign key_bcoeffs and write output")
# here we use only 16 key cycles

for j in range(16) :

    print("j value:  ", j) 
    key_bcoeffs = torch.zeros(num_keys, n)  # 16 by 40
    for k in range(16) :   # was range(num_labels) can also change to other ranges like range(1,17) etc.
        file = "dulcimerA3-f/bcoeffs/bcoeffs-n40-seg" +str(labels[j+k][0]) + "-cyc" +str(labels[j+k][1]) + ".txt"
        print("file:  ", file)
        bcoeffs = import_bcoeffs(file)
        key_bcoeffs[k] = 3.0 * gains[k] * bcoeffs

    # print("key_bcoeffs:")
    # print(key_bcoeffs)
    
    print("\n")
    print("------------------------------------------------------------------------")
    print("\n")



