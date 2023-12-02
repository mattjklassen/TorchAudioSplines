
# ----- Brief Description -----
# 
# create melody based on spline curve, also with varying note durations
# but now use the stationary points to determine pitch and note duration.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# use notes on the one eighth second, with pitches obtained from spline.
# Initially use spline generated randomly with torchSpline.py
# pitches can be determined by starting at y=0 being A220, and y=1 A440, y=-1 A110.
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
from getStatVals import getSplineVals, getStatPts
from getKnots import getKnots

# main part of script

# use insertWavTone to put each tone of melody and write to wav file
# also, use different time values for notes based on spline
# for this version we will import the stationary points and use those to
# generate both pitches and time values.  

notes = 0     # number of notes in melody (we don't know yet)
f0 = 220.0    # f0 = fundamental frequency of first note
time0 = 0.5  # time0 = duration of first note in seconds (use to scale other durations)

file = "bcoeffs0.txt"
bcoeffs = import_bcoeffs(file)
n = bcoeffs.size(dim=0)
print("n = ", n)

# in order to compute lengths for arrays of note times and frequencies, we need the
# array of stationary points to use for those pitches and durations.

knotVals = getKnots(n)
# get the spline values over [0,1] at 1000 points
new_splineVals = getSplineVals(bcoeffs, knotVals, 1000)
# search through those 1000 points for stationary points, and also include endpoints
new_statPts = getStatPts(new_splineVals)
num_statPts = len(new_statPts)
notes = num_statPts - 1

# here we are setting the keys, or indices of key cycles
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

# new_statPts is a list of pts [x,y] so we can use the 2^y in the usual way for 
# pitches and then get durations from x[i] - x[i-1], i=1 ... num_statPts - 1. 
# Those durations can then be scaled to make the first duration a chosen target value.
# Pitches are determined by single values, durations are determined as subintervals.
# So first pitch could be y[0]=0, whereas first duration could be x[1]-x[0], or x[1].
# A melody could use the last pitch same as first given by y[1]=0, or not.

time1 = time0
sample_rate = 44100.0
start_time = 0.0
note_times = torch.zeros(num_statPts)
frequencies = torch.zeros(num_statPts)
interp_method = 1
gains = torch.tensor([0.7,1.0,1.0])

key_bcoeffs = torch.zeros(num_keys, n)
for i in range(num_keys) :
    key_bcoeffs[i] = bcoeffs

x = 0.0
y = 0.0
t = 0.0
time2 = 0

# set note_times (durations), keys, and fundamental frequencies with spline values
# note i starts at time x[i] with pitch 2^y[i] and goes through time x[i+1]
for i in range(notes) :
    print("computing values for i = ", i)
    x0 = new_statPts[i][0]
    x1 = new_statPts[i+1][0]
    x = new_statPts[i][1]
    x *= 4.0
    print("x (spline) value scaled by 4: ", x)
    y = np.exp2(x)
    print("y (exp2) value: ", y)
    note_times[i] = x1 - x0
    print("note duration: ", note_times[i])
    frequencies[i] = f0 * y
    print("frequencies[i]: ", frequencies[i])
    temp = torch.tensor(keys0)
    temp = y * temp 
    temp = note_times[i] * temp
    for j in range(num_keys) :
        keys[j] = int(temp[j])
    note_keys[i] = torch.tensor(keys)
    print("keys for i = ", i, keys)
    print("")

time2 = time0 / note_times[0]
temp = torch.tensor(note_times)
temp *= time2
for j in range(notes) :
    note_times[j] = temp[j]
print("total time: ", time2)

# time2 = duration of entire waveform or melody in seconds
# these are different times computed from the stationary points of spline
waveform_length = int(time2 * sample_rate) + 10  # in samples
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



