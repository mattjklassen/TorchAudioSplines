
# ----- Brief Description -----
# 
# (based on testGenWav.py which was to construct one tone of 1 sec long with genWavTone() )
# This one is for testing a dulcimer tone using 32 key cycles, with various cases.
# The key cycles are chose from a 3 second long sample called dulcimerA3-f.wav with
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
time = 10.0
# time = 0.25
sample_rate = 44100.0
# n = bcoeffs.size(dim=0)
n = 40
# keys = torch.tensor([0,10,20,30,50,70,90])
keys = torch.tensor([0,1,2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,110,120,130,140,150,160,170,190,210])
print("keys: ", keys)
num_keys = keys.size(dim=0)
print("num_keys:  ", num_keys)

# this key sequence works for one second at frequency 220 Hz
# for shorter time we can scale this as:
temp = time * keys
print("temp keys: ", temp)
for i in range(num_keys) :
    keys[i] = int(temp[i])
print("keys: ", keys)

gains = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
for j in range(32) :
    if j > 12 :
        scale = 1.0 - float(j - 12) / 20
        gains[j] *= scale

print("scaled gains:")
print(gains)

num_gains = len(gains)
# print("num_gains: ", num_gains)

interp_method = 1
key_bcoeffs = torch.zeros(num_keys, n)  # say 32 by 40

# key cycles chosen from 16 segments: 0 through 10,20,30,40,50,60. 
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

print("assigning key_bcoeffs")
for k in range(num_labels) :
    file = "dulcimerA3-f/bcoeffs/bcoeffs-n40-seg" +str(labels[k][0]) + "-cyc" +str(labels[k][1]) + ".txt"
    print("file:  ", file)
    bcoeffs = import_bcoeffs(file)
    key_bcoeffs[k] = 1.5 * gains[k] * bcoeffs

wav_data = genWavTone(f0, time, sample_rate, key_bcoeffs, keys, gains, interp_method)

size_out = int(sample_rate * time)
waveform = torch.empty(1, size_out)
for i in range(size_out) :
    waveform[0,i] = wav_data[i]

print("we have wav data")
print("now writing wav file: audio/tone5.wav")

path = "../audio/tone5.wav"
torchaudio.save(
    path, waveform, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)

path = "../audio/tone5.wav"
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)



