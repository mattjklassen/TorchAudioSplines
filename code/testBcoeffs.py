# ----- Brief Description -----
# 
# get bcoeffs (B-spline coefficients) from a cycle (or segment) [a,b] in a waveform
# inputs: waveform (tensor of audio data), cycle = [a,b], n = dimension of splines
# return: bcoeffs vector 
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Note: we are not assuming the endpoint values are zero, so also the bcoeffs c[0]
# and c[n-1] are also not necessarily zero.
#
# ----- ----- ----- ----- -----


import torch
import torchaudio
import numpy as np
import math
from computeBsplineVal import newBsplineVal, computeSplineVal 
from getBcoeffs import getBcoeffs, import_bcoeffs, export_bcoeffs


# assume waveform comes from read of this type: 
waveform, sample_rate = torchaudio.load("../audio/input.wav")
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
segments = torch.tensor_split(waveform, 16, dim=1)
waveform = segments[2]

np_waveform = waveform.numpy() 
num_channels, num_frames = np_waveform.shape
# i^th sample value is now np_waveform[0,i] (but becomes data below)
# number of samples = num_frames
    
n = 15  # dimension of cubic splines
a = 500.1
b = 600.45
cycle = [a, b]

print("waveform.shape:  ", waveform.shape)
print(waveform)
print("np_waveform.shape:  ", np_waveform.shape)
print(np_waveform)
data = torch.squeeze(waveform).numpy()
print("data.shape:  ", data.shape)
print(data)

bcoeffs = getBcoeffs(waveform, sample_rate, cycle, n)
print("bcoeffs from: input.wav")
print("with interval: ", cycle)
print("and dimension n = ", n)
print(bcoeffs)
print("shape: ", bcoeffs.shape)
np_bcoeffs = bcoeffs.numpy()
print("np shape: ", np_bcoeffs.shape)
print(np_bcoeffs)

bcoeffs_str = []
for i in range(n) :
    print(np_bcoeffs[i])
    bcoeffs_str.append(str(np_bcoeffs[i]))
    bcoeffs_str.append('\n')

with open('bcoeffs0.txt', 'w') as f:
    f.writelines(bcoeffs_str)
    f.close()

print("reading bcoeffs0.txt")
bcoeffs_str = []
with open('bcoeffs0.txt', 'r') as f:
    bcoeffs_str = f.readlines()
    f.close()

bcoeffs = []
for i in range(len(bcoeffs_str)) :
    bcoeffs.append(float(bcoeffs_str[i]))

print(bcoeffs)
