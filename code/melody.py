
# ----- Brief Description -----
# 
# create melody based on spline curve
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# use notes on the one quarter or eighth second, with pitches obtained from spline.
# Initially use spline generated randomly with torchSpline.py
# pitches can be determined by starting at y=0 being A 220, and y=1 A 440, y=-1 A110.
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
from computeBsplineVal import computeSplineVal

# main part of script

# use insertWavTone to put each tone of melody and write to wav file
# next, need to use different time values for notes based on spline

f0 = 82.5  
notes = 16
# time1 = duration of one note = 1/8 second
time1 = 0.25       
time1 = 0.125       
# time2 = duration of entire waveform or melody
time2 = time1 * notes # 50 notes, 1/8 second each, 0.125 * 50 = 6.25 sec
sample_rate = 44100.0
waveform_length = int(time2 * sample_rate)  # = 275,625 samples
waveform = torch.zeros(waveform_length)
tone_length = time1 * sample_rate   # length of one tone in samples
# start_time in samples for insertion of current note into waveform
start_time = 0.0

file = "bcoeffs0.txt"
bcoeffs = import_bcoeffs(file)
n = bcoeffs.size(dim=0)
# keys = torch.tensor([0,10,20,30,50,70,90])
# the following keys work for f0 = 110, and time1 = 0.125
keys = torch.tensor([0,30,90])
num_keys = keys.size(dim=0)

# scale keys by frequency ratio f0/110
temp = (f0 / 110) * keys
for i in range(num_keys) :
    keys[i] = int(temp[i])
print("keys: ", keys)

# scale keys by time1
temp = time1 * keys
print("temp keys: ", temp)
for i in range(num_keys) :
    keys[i] = int(temp[i])
print("keys: ", keys)

# gains = torch.tensor([1,1,1,1,1,1,1])
# gains = torch.tensor([0.7,0.8,0.9,1.0,1.0,1.0,1.0])
gains = torch.tensor([0.7,1.0,1.0])
interp_method = 1
key_bcoeffs = torch.zeros(num_keys, n)
for i in range(num_keys) :
    key_bcoeffs[i] = bcoeffs

y = 0.0
t = 0.0
f = f0

for i in range(notes) :
    print("inserting for i = ", i)
    insertWavTone(waveform, start_time, f, time1, sample_rate, key_bcoeffs, keys, gains, interp_method)
    # first tone is at f0 = 110, next will be determined from spline as cent value 1200*y
    # so desired frequency f = f0 * 2^y = f0 * exp2(y) 
    start_time += tone_length
    t = start_time / waveform_length
    print("t value: ", t)
    x = computeSplineVal(3, n-3, bcoeffs, t)
    x *= 2
    print("x (spline) value: ", x)
    y = np.exp2(x)
    print("y (exp2) value: ", y)
    f = f0 * y
    temp_keys = y * temp
    for i in range(num_keys) :
        keys[i] = int(temp_keys[i])
    print("temp_keys: ", temp_keys)
    print("keys: ", keys)

waveform_out = torch.unsqueeze(waveform, dim=0)

print("we have wav data")
print("now writing wav file")

path = "../audio/melody.wav"
torchaudio.save(
    path, waveform_out, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)

path = "../audio/melody.wav"
file_waveform, sample_rate = torchaudio.load(path)
np_waveform = file_waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)



