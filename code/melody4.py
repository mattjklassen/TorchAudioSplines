
# ----- Brief Description -----
# 
# Create melody based on spline curve using spline values y to determine pitch
# and x values to determine time durations.  If notes=0 then we use stationary points
# and if notes>0 we use that many equal divisions of the interval [0,1].
# Durations are scaled so that first note lasts for time0 seconds.
# possible command line: (see below for details)
# python melody4.py bcoeffs0.txt f0=234 scale=3 notes=4 shift=5 time0=0.123 r i
# or to use stationary points and do retrograde inversion (and other defaults):
# python melody4.py bcoeffs0.txt r i
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# inversion of intervals can be realized in our system on the y axis
# by simplying taking the negative of spline values before applying 2^y, 
# then shifting by some constant (say 1 for within the octave).
# For example, if the original values are in [-1,1] then the values
# 0 and 7/12 map to a musical interval of an equal tempered perfect fifth,
# so the classical inversion of this would be to change the sign and shift
# up by 1.  We can generalize by shifting by any number up or down.
# Retrograde of a melody is to run it backwards in time.  This is simple
# and visually can be seen by just running the graph from right to left.
#
# to allow for transformations we need arguments on command line:
# r (retrograde) i (inversion) 
# as well as float parameters:  f0 = starting frequency (default 220)
# and scalar value "scale" for spline values on y axis before taking exp2(),
# and shift value "shift" for spline values on y axis before taking exp2().
# and also notes = number of notes to create from spline values as melody
# if notes=0 then we use stationary points instead
# if notes>0 then we use regularly spaced notes over interval [0,1]
# and time0=0.125 is duration of first note = 1/8 second
#
# for example, shift=1 would change y-values from [-1,1] to [0,2]
# so together with inversion this maps y to 1-y.  This has the effect
# of the classical inversion of intervals, since a fifth y=7/12
# would map to a fourth y=5/12. Of course, without the shift, this
# would still be a fourth but in the octave below, same pitch class.
#
# args: (first two are forced, all others are optional and have defaults)
# [0] melody4.py
# [1] bcoeffs-file.txt
# knots-file.txt  (default is computed as 0,0,0,0,1/k,...,(k-1)/k,1,1,1,1)
# note: knots-file must contain the string "knots" but not the string "="
# for example [2] knots-0.txt
# invert=0    (default is 0) (this is set to 1 if "i" is on command line)
# retro=0     (default is 0) (this is set to 1 if "r" is on command line)
# f0=220      (default is 220)
# scale=1     (default is 1)
# shift=0     (default is 0)
# notes=0     (default is 0)
# time0=0.5   (default is 0.5)
#
# ----- ----- ----- ----- -----


import sys
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from argMaxSpec import plotSpecArgMax, getArgMax
from cycleSpline import plotCycleSpline
from getCycles import getCycles
from getBcoeffs import import_bcoeffs, export_bcoeffs
from genWavTone import genWavTone, insertWavTone2, insertWavTone
from computeBsplineVal import computeSplineVal
from getStatVals import getSplineVals, getStatPts
from getKnots import getKnots, import_knots

# default params 
bcoeffs_file = "bcoeffs0.txt" 
invert = 0
retro = 0
f0 = 220
init_f0 = 220
scale = 1
shift = 0
notes = 0
time0 = 0.25
special_knots = 0

print("Argument List:", str(sys.argv))
args = len(sys.argv)
if args < 2 :
    print("need bcoeffs file as second command line arg")
    sys.exit(0)

bcoeffs_file = sys.argv[1]
bcoeffs = import_bcoeffs(bcoeffs_file)
n = bcoeffs.size(dim=0)
print("n = ", n)

if args > 2 :
    for i in range(2,args) :
        arg = sys.argv[i]
        if "knots" in arg :
            knots_file = arg
            special_knots = 1
        if arg == "i" :
            invert = 1
            print("setting inversion on")
        if arg == "r" :
            retro = 1
            print("setting retrograde on")
        if "=" in arg :
            index = arg.find("=")
            param = arg[:index]
            val = float(arg[index+1:])
            print("param:", param, " value: ", val)
        if param == "f0" :
            f0 = val
        if param == "scale" :
            scale = val
        if param == "shift" :
            shift = val
        if param == "notes" :
            notes = int(val)
        if param == "time0" :
            time0 = val

print("params: ")
print("f0:  ", f0)
print("scale:  ", scale)
print("shift:  ", shift)
print("notes:  ", notes)
print("time0:  ", time0)
print("invert:  ", invert)
print("retro:  ", retro)

# if notes == 0 then we use stationary points
# in order to compute lengths for arrays of note times and frequencies, we need the
# array of stationary points to use for those pitches and durations.

knotVals = getKnots(n)
if special_knots == 1 :
    knotVals = import_knots(knots_file)

# get the spline values over [0,1] at 1000 points
# (this was lifted from previous code use to graph the spline with matplot)
new_splineVals = getSplineVals(bcoeffs, knotVals, 1000)

# search through those 1000 points for stationary points, and also include endpoints
new_statPts = getStatPts(new_splineVals)
num_statPts = len(new_statPts)
notes = num_statPts - 1  # since note[i] will have duration x[i+1]-x[i]
# and frequency given by 2^y[i]*f0, so in the forward case we have each
# note determined by subinterval with pitch computed at the start of subinterval
# working left to right, and for retrograde this is just flipped to go right to left.
# note: notes = number of subintervals
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

# NEED TO ADD MORE BCOEFFS HERE for different key cycles and interpolation

x = 0.0
y = 0.0
t = 0.0
time2 = 0

# next set note_times (durations), keys, and fundamental frequencies with spline values
# note i starts at time x[i] with pitch 2^y[i] and goes through time x[i+1]

# next loop goes from 0 to notes-1 and computes durations as subinterval lengths on x axis
# running forwards (if retro == 0) or backwards (if retro == 1).
# but the pitch should still be computed at the left endpoint of subinterval

print("new_statPts:")
print(new_statPts)

too_close = 1

while too_close == 1 :
    too_close = 0
    notes = len(new_statPts) - 1
    print("notes = ", notes)
    for i in range(notes) :
        x0 = new_statPts[i][0]
        x1 = new_statPts[i+1][0]
        if x1 - x0 < 0.02 :
            print("too close:  ", i, ": ", x0, i+1, ": ", x1, "diff:  ", x1-x0)
            too_close = 1
            to_remove = i+1
    if too_close == 1 :    
        print("removing stat point: ", to_remove)
        del new_statPts[to_remove]
        print("new_statPts after removal:")
        print(new_statPts)

for i in range(notes) :
    print("computing values for i = ", i)
    x0 = new_statPts[i][0]
    x1 = new_statPts[i+1][0]
    y = new_statPts[i][1]
    if retro == 1 :
        x0 = new_statPts[notes-i-1][0]
        x1 = new_statPts[notes-i][0]
        y = new_statPts[notes-i-1][1]
        print("x0: ", x0, "x1: ", x1)
    if invert == 1 :
        y *= -1
        print("y value is inverted: ", y)
    y *= scale
    print("y value is scaled by ", scale, ":  ", y)
    y += shift
    print("y value is shifted by ", shift, ":  ", y)
    y = np.exp2(y)
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

# sys.exit(0)

# scale first subinterval to have length time0:
time2 = time0 / note_times[0]
# for retrograde the durations are reversed, so to keep same total length use last:
if retro == 1 :
    time2 = time0 / note_times[notes-1]
temp = torch.tensor(note_times)
# scale all subintervals (note-times) by time2:
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
    if i > 0 :
        ratio = frequencies[i] / frequencies[i-1]
        centval = (1200 / np.log(2)) * np.log(ratio)
        print("interval:  ratio = ", ratio, " cent value =  ", centval)
        ratio = frequencies[i] / init_f0 
        centval = (1200 / np.log(2)) * np.log(ratio)
        print("cent value relative to initial f0:  ", centval)
    print("note_keys[i]: ", note_keys[i])
    # insertWavTone(waveform, start_time, f0, time1, sample_rate, key_bcoeffs, note_keys[i], gains, interp_method)
    insertWavTone2(waveform, start_time, f0, time1, sample_rate, key_bcoeffs, knotVals, note_keys[i], gains, interp_method)
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








