
# ----- Brief Description -----
# 
# create melody based on spline curve, also with varying note durations
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# use notes on the one eighth second, with pitches obtained from spline.
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
# also, use different time values for notes based on spline

notes = 40     # number of notes in melody (sequence of notes)
f0 = 110.0    # f0 = fundamental frequency of first note
time0 = 0.25  # time0 = duration of first note = 1/2 second

keys_float = [0.0,30.0,90.0]  # placeholder keys for f0 = 110, time = 1 sec
num_keys = len(keys_float)
temp = torch.tensor(keys_float)
temp *= time0
keys0 = []
for i in range(num_keys) :
    keys0.append(int(temp[i]))   # keys0 = keys for first note with f0 and time0
note_keys = torch.zeros(notes, num_keys)
keys = []
for i in range(num_keys) :
    keys.append(int(keys_float[i]))

time1 = time0
sample_rate = 44100.0
start_time = 0.0
note_times = torch.zeros(notes)
frequencies = torch.zeros(notes)
interp_method = 1
gains = torch.tensor([0.7,1.0,1.0])

file = "bcoeffs1.txt"
bcoeffs = import_bcoeffs(file)
n = bcoeffs.size(dim=0)

key_bcoeffs = torch.zeros(num_keys, n)
for i in range(num_keys) :
    key_bcoeffs[i] = bcoeffs

x = 0.0
y = 0.0
t = 0.0
time2 = 0

# set note_times (durations), keys, and fundamental frequencies with spline values
for i in range(notes) :
    print("computing note_times[i] for i = ", i)
    t = float(i) / float(notes)
    print("t value: ", t)
    x = computeSplineVal(3, n-3, bcoeffs, t)
    print("x (spline) value: ", x)
    y = np.exp2(x)
    print("y (exp2) value: ", y)
    note_times[i] = time0 * y
    # can change * to / for inverse of note duration time 
    # note_times[i] = time0 / y
    print("note_times[i]: ", note_times[i])
    time2 += note_times[i]
    print("time2: ", time2)
    frequencies[i] = f0 * y
    print("frequencies[i]: ", frequencies[i])
    temp = torch.tensor(keys0)
    temp = y * y * temp
    for j in range(num_keys) :
        keys[j] = int(temp[j])
    note_keys[i] = torch.tensor(keys)
    print("keys for i = ", i, keys)
    print("")

print("note_keys: ", note_keys)

# time2 = duration of entire waveform or melody in seconds
# these are now different times computed on the spline, but were previously:
# time2 = time1 * notes # 50 notes, 1/8 second each, 0.125 * 50 = 6.25 sec
waveform_length = int(time2 * sample_rate)  # in samples
waveform = torch.zeros(waveform_length)

x = 0.0
y = 0.0
t = 0.0
f = f0

for i in range(notes) :
    print("inserting for i = ", i)
    time1 = note_times[i]
    f0 = frequencies[i]
    print("start_time, time1, f0:  ", start_time, time1, f0)
    print("note_keys[i]: ", note_keys[i])
    insertWavTone(waveform, start_time, f0, time1, sample_rate, key_bcoeffs, note_keys[i], gains, interp_method)
    start_time += time1 * sample_rate

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



