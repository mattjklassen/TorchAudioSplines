
# ----- Brief Description -----
# 
# write two octave chromatic scale to wav file
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# use notes on the one eighth second
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
from genWavTone import genWavTone, insertWavTone

# use insertWavTone to put each tone of ET chromatic scale in tensor and write to wav file

# main part of script

# comments from scale.py
# From micro to macro scales, we have:
# time0 = one cycle
# time1 = one note
# time2 = one melody

f0 = 55.0  
notes = 25
time1 = 0.125       # duration of one note = 1/8 second
time2 = 0.125 * notes # 12 notes of chromatic scale in each octave, plus last note 2 octaves,
                    # 1/8 second each, 0.125 * 25 = 3.125 sec
sample_rate = 44100.0
waveform_length = int(time2 * sample_rate)  # = 137,812 samples
waveform = torch.zeros(waveform_length)
tone_length = 0.125 * sample_rate

file = "bcoeffs1.txt"
bcoeffs = import_bcoeffs(file)
n = bcoeffs.size(dim=0)
# keys = torch.tensor([0,10,20,30,50,70,90]) # for f0 = 110
keys = torch.tensor([0,5,10,15,25,35,45])  # for f0 = 55
print("keys: ", keys)
num_keys = keys.size(dim=0)
# this key sequence works for one second at frequency 100 Hz

temp = time1 * keys
print("temp keys: ", temp)
for i in range(num_keys) :
    keys[i] = int(temp[i])
print("keys: ", keys)

# gains = torch.tensor([1,1,1,1,1,1,1])
gains = torch.tensor([0.7,0.8,0.9,1.0,1.0,1.0,1.0])
interp_method = 1
key_bcoeffs = torch.zeros(num_keys, n)
for i in range(num_keys) :
    key_bcoeffs[i] = bcoeffs

start_time = 0.0

for i in range(25) :
    insertWavTone(waveform, start_time, f0, time1, sample_rate, key_bcoeffs, keys, gains, interp_method)
    f0 *= 1.059463
    start_time += tone_length
    temp *= 1.059463 
    for i in range(num_keys) :
        keys[i] = int(temp[i])
    print("keys: ", keys)
    print("temp: ", temp)

waveform_out = torch.unsqueeze(waveform, dim=0)

print("we have wav data")
print("now writing wav file")

path = "../audio/scale.wav"
torchaudio.save(
    path, waveform_out, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)

path = "../audio/scale.wav"
file_waveform, sample_rate = torchaudio.load(path)
np_waveform = file_waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)



