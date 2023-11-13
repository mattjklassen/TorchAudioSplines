
# ----- Brief Description -----
# 
# construct one tone of 1 sec long with genWavTone2()
# uses bcoeffs and knotVals to allow for new knot sequence like splinusoid
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
from genWavTone import genWavTone2

# test the function genWavTone2.py with waveform data and write outputs to wav and pdf
# genWavTone2(f0, time, sample_rate, key_bcoeffs, knotVals, keys, gains, interp_method) 

# main part of script

f0 = 110.0
# f0 /= 1.632526919
# f0 *= 1.059463
# f0 *= 1.059463
time = 1.0
# time = 0.25
sample_rate = 44100.0
file = "bcoeffs-sin2.txt"
bcoeffs = import_bcoeffs(file)
n = bcoeffs.size(dim=0)
d = 3
k = 8
N = n + d   # N = 17
keys = torch.tensor([0,10,20,30,50,70,90])
print("keys: ", keys)
num_keys = keys.size(dim=0)
# this key sequence works for one second at frequency 100 Hz
# for shorter time we can scale this as:
temp = time * keys
print("temp keys: ", temp)
for i in range(num_keys) :
    keys[i] = int(temp[i])
print("keys: ", keys)

gains = [0.2,0.6,0.9,1.0,1.0,0.8,0.6]
interp_method = 1
key_bcoeffs = torch.zeros(num_keys, n)

print("assigning key_bcoeffs")
for i in range(num_keys) :
    key_bcoeffs[i] = gains[i] * bcoeffs

knotVals = torch.zeros(N+1)

incr = 1/k

for i in range(4) :
    j = i + 7
    knotVals[j] = 1/2
    j = i + 14
    knotVals[j] = 1

for i in range(3) :
    j = i + 4
    knotVals[j] = knotVals[j-1] + incr
    j = i + 11
    knotVals[j] = knotVals[j-1] + incr

print("knotVals: ")
print(knotVals)

wav_data = genWavTone2(f0, time, sample_rate, key_bcoeffs, knotVals, keys, gains, interp_method)

size_out = int(sample_rate * time)
waveform = torch.empty(1, size_out)
for i in range(size_out) :
    waveform[0,i] = wav_data[i]

print("we have wav data")
print("now writing wav file: audio/tone.wav")

path = "../audio/tone.wav"
torchaudio.save(
    path, waveform, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)

path = "../audio/tone.wav"
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)



