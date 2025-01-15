
# ----- Brief Description -----
# 
# construct one polyphonic tone of 2 sec long with genWavTone()
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

f0 = 50.0
time = 1.0
sample_rate = 44100.0

file0 = "poly_tones/4-5-6-7/poly_bcoeffs.txt"
file1 = "poly_tones/5-6-7-8/poly_bcoeffs.txt"
file2 = "poly_tones/6-7-8-10/poly_bcoeffs.txt"
file3 = "poly_tones/7-8-10-12/poly_bcoeffs.txt"

bcoeffs0 = import_bcoeffs(file0)
bcoeffs1 = import_bcoeffs(file1)
bcoeffs2 = import_bcoeffs(file2)
bcoeffs3 = import_bcoeffs(file3)

n = bcoeffs1.size(dim=0)

keys = torch.tensor([0,10,20,30])
print("keys: ", keys)
num_keys = keys.size(dim=0)
# this key sequence works for one second at frequency 100 Hz
# for shorter time we can scale this as:
temp = time * keys
print("temp keys: ", temp)
for i in range(num_keys) :
    keys[i] = int(temp[i])
print("keys: ", keys)

gains = [0.8,1.0,0.6,0.3]
interp_method = 1
key_bcoeffs = torch.zeros(num_keys, n)

# here we assign the bcoeffs to each row of key_bcoeffs

print("assigning key_bcoeffs")
key_bcoeffs[0] = gains[0] * bcoeffs0
key_bcoeffs[1] = gains[1] * bcoeffs1
key_bcoeffs[2] = gains[2] * bcoeffs2
key_bcoeffs[3] = gains[3] * bcoeffs3

wav_data = genWavTone(f0, time, sample_rate, key_bcoeffs, keys, gains, interp_method)

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

# retrieve and check
path = "../audio/tone.wav"
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)



