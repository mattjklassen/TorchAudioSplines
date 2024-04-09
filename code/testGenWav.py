
# ----- Brief Description -----
# 
# construct one tone of 1 sec long with genWavTone()
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

f0 = 110.0
# f0 *= 1.059463
# f0 *= 1.059463
time = 0.25
# time = 0.25
sample_rate = 44100.0
file = "bcoeffs1.txt"
bcoeffs = import_bcoeffs(file)
n = bcoeffs.size(dim=0)
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

gains = [0.8,0.9,1.0,0.8,0.5,0.3,0.1]
interp_method = 1
key_bcoeffs = torch.zeros(num_keys, n)

# here we assign the same bcoeffs to each row of key_bcoeffs, so all keys are the same spline
# which defeats the purpose of cycle interpolation.
print("assigning key_bcoeffs")
for i in range(num_keys) :
    key_bcoeffs[i] = gains[i] * bcoeffs

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

path = "../audio/tone.wav"
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)



