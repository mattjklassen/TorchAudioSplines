
# ----- Brief Description -----
# 
# melody6.py is derived from melody5.py but now adding polyphony or voicing.
#
# 1. use config file mel6config.txt which contains the parameters used
#    to construct the melody or melodic fragment from bcoeffs files etc.
#
# 2. add polyphony, or voicing:
#    in addition to those configs in mel5config.txt we now control the duration
#    of notes in various ways.  For instance, note durations can all be set to
#    last for twice as long as the designated duration in the melody, so they overlap.
#    The waveform for each note will then be stored in a buffer chosen from a list of 
#    buffers, each representing a voice, or channel, and then these buffers will all 
#    be mixed before writing the final output.  
#
# ----- ----- ----- ----- -----

# To Do:
#
# Put in the option to specify first interval in cents.
# This can be done by calculating scale to achieve specified first interval.
# For example, say initial f0 = 220 and second note occurs at y_1 = 0.3 
# and we want first interval to be a just perfect fifth.
# Then we can accomplish this by scaling the y axis so that 2^(y_1) = 3/2.
# So y_1 = log2(1.5) and we scale the signal y-values by log2(1.5) / 0.3.
#
# Make the calculation of initial f0 reflect the use of transposition or shift.
#

# ------- More Details --------
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
from genWavTone import genWavTone, insertWavTone, insertWavTone2, insertWavTone3
from computeBsplineVal import computeSplineVal
from getStatVals import getSplineVals, getStatPts
from getKnots import getKnots, import_knots

# first need to parse mel6config.txt, main difference from mel5 is voices

file = "mel6config.txt"

reading_bcoeffs = False
reading_key_knots = False
reading_keys = False
reading_mel_bcoeffs = False
reading_mel_knots = False
reading_params = False
reading_gains = False
reading_audio_source = False
key_bcoeffs_defined = False
key_knots_defined = False
mel_knots_defined = False

n = 0
j = 0
num_keys = 0
keys = []
bcoeffs_array = []
gains = []
sample_rate = 44100
interp_method = 1
f0 = 220
notes = 0
stat = 1
time0 = 0.25
invert = 0
retro = 0
scale = 1
shift = 0
audio_source = ""
voices = 0
voice_scalar = 1

with open(file, 'r') as f:
    config_lines = f.readlines()
    f.close()

N_config = len(config_lines)

for i in range(N_config) :
    # parse config_lines to find and set params and read keys, bcoeffs and knots
    line = config_lines[i]
    if "#" in line :
        continue
    if "KEY BCOEFFS" in line :
        reading_bcoeffs = True
        continue
    if "KEY KNOTS" in line :
        reading_key_knots = True
        continue
    if "KEYS" in line :
        reading_keys = True
        continue
    if "KEY GAINS" in line :
        reading_gains = True
        continue
    if "MELODIC CONTOUR" in line :
        reading_mel_bcoeffs = True
        continue
    if "MEL KNOTS" in line :
        reading_mel_knots = True
        continue
    if "AUDIO SOURCE" in line :
        reading_audio_source = True
        print("set reading_audio_source = True")
        continue
    if "PARAMS" in line :
        reading_params = True
        continue

    # keys are required in config
    if reading_keys :
        if line == '\n' :
            reading_keys = False
            num_keys = len(keys)
            j = 0
        else :
            new_key = float(line)
            keys.append(new_key)
            j += 1

    # key bcoeffs are required in config
    if reading_bcoeffs :
        if "bcoeffs" in line :
            bcoeffs_file = line.rstrip()
            bcoeffs = import_bcoeffs(bcoeffs_file)
            n = bcoeffs.size(dim=0)
            # print("imported from file: ", bcoeffs_file, " with n = ", n)
            if not key_bcoeffs_defined :
                key_bcoeffs_defined = True
            bcoeffs_array.append(bcoeffs)
            j += 1
        else :
            reading_bcoeffs = False      
            j = 0

    # key gains are optional in config
    if reading_gains :
        if line == '\n' :
            reading_gains = False
            j = 0
        else :
            # print("reading gains")
            new_gain = float(line)
            gains.append(new_gain)
            j += 1

    # knots are optional in config, so blank line signals to move on
    if reading_key_knots :
        if line == '\n' :
            reading_key_knots = False
            continue
        else :
            # print("reading key knots")
            if "knots" in line :
                knots_file = line.rstrip()
                key_knots = import_knots(knots_file)
                # print("imported from file: ", knots_file)
                reading_key_knots = False
                continue

    # this file read is forced, so must be present in the config
    if reading_mel_bcoeffs :
        # print("reading mel_bcoeffs")
        if "bcoeffs" in line :
            mel_bcoeffs_file = line.rstrip()
            mel_bcoeffs = import_bcoeffs(mel_bcoeffs_file)
            n_mel = mel_bcoeffs.size(dim=0)
            # print("imported from file: ", mel_bcoeffs_file, " with n_mel = ", n_mel)
            reading_mel_bcoeffs = False
            continue

    # knots are optional in config, so blank line signals to move on
    if reading_mel_knots :
        if line == '\n' :
            reading_mel_knots = False
            continue
        else :
            # print("reading mel knots")
            if "knots" in line :
                mel_knots_file = line.rstrip()
                mel_knots = import_knots(mel_knots_file)
                # print("imported from file: ", mel_knots_file)
                reading_mel_knots = False
                # print("mel_knots size: ", len(mel_knots))
                # print("n_mel + 4 : ", n_mel + 4)
                continue

    # audio source file is optional in config, so blank line signals to move on
    # if source file is given, then this will be used when writing output as follows:
    # audio_prefix/melody-<transform>.wav where transform is defined as:
    # transform = prime if invert=0 and retro=0
    # transform = inversion if invert=1 and retro=0
    # transform = retrograde if invert=0 and retro=1
    # transform = retrograde-inversion if invert=1 and retro=1
    if reading_audio_source :
        if line == '\n' :
            reading_audio_source = False
            continue
        else :
            if "wav" in line :
                audio_source = line.rstrip()
                print("found audio_source : ", audio_source)
                continue

    # params can be set in config, or left as the following defaults:
    # f0=220 notes=0 stat=1 time0=0.25 invert=0 retro=0 scale=1 shift=0
    if reading_params :
        if line == '\n' :
            # reading_params = False
            param = ""
            j = 0
        else :
            # print("reading params")
            if "=" in line :
                index = line.find("=")
                param = line[:index]
                val = float(line[index+1:])
                # print("param:", param, " value: ", val)
            if param == "sample_rate" :
                sample_rate = val
            if param == "interp_method" :
                interp_method = val
            if param == "f0" :
                f0 = val
            if param == "notes" :
                notes = int(val)
            if param == "stat" :
                stat = int(val)
            if param == "time0" :
                time0 = val
            if param == "invert" :
                invert = int(val)
            if param == "retro" :
                retro = int(val)
            if param == "scale" :
                scale = val
            if param == "shift" :
                shift = val
            if param == "voices" :
                voices = int(val)
            if param == "voice_scalar" :
                voice_scalar = val
            j += 1
            # print("j is now: ", j)

if shift > 0 or shift < 0 :
    f0 = f0 * np.exp2(shift)
    shift = 0
    print("shifted f0:  ", f0)

transform = "prime"
if invert == 1 and retro == 0 :
    transform = "inversion"
if invert == 0 and retro == 1 :
    transform = "retrograde"
if invert == 1 and retro == 1 :
    transform = "retrograde-inversion"

print("transform:  ")
print(transform)
print("")

print("keys: ")
print("")
print(keys)
print("")

while len(gains) < num_keys :
    gains.append(gains[-1])
    
key_gains = torch.tensor(gains)
print("key_gains: ")
print("")
print(key_gains)
print("")

key_bcoeffs = torch.zeros(num_keys, n)
for m in range(num_keys) :
    temp = torch.tensor(bcoeffs_array[m])
    key_bcoeffs[m] = temp

# print("key_bcoeffs updated from array: ")
# print("")
# print(key_bcoeffs)
# print("")

if not key_knots_defined :
    key_knots = getKnots(n)
# print("key_knots: ")
# print("")
# print(key_knots)
# print("")

# print("mel_bcoeffs: ")
# print("")
# print(mel_bcoeffs)
# print("")

if not mel_knots_defined :
    mel_knots = getKnots(n_mel)
# print("mel_knots: ")
# print("")
# print(mel_knots)
# print("")

print("params: ")
print("f0:  ", f0, "     fundamental frequency of first note")
print("notes:  ", notes, "      number of notes in melody")
print("stat:  ", stat, "       1 means use stationary points")
print("time0:  ", time0, "   time value of first note in seconds")
print("invert:  ", invert, "     1 means do inversion")
print("retro:  ", retro, "      1 means do retrograde")
print("scale:  ", scale, "    1 means no scaling of y axis")
print("shift:  ", shift, "    0 means no shifting of y axis")
print("interp_method:  ", interp_method, "     1 = linear interpolation of bcoeffs")
print("sample_rate:  ", sample_rate, "     for output wav file")

# print("j is now: ", j)
print("")

# if stat == 1 then we use stationary points of melodic contour spline for notes,
# otherwise we use regularly spaced notes which are sampled from melodic contour spline

# first form the melodic contour spline with mel_bcoeffs and mel_knots (or default knots)
# we can keep the waveform splines separate from melodic contour spline if desired, which
# also include having different dimensions for each, say n for waveform and n_mel for mel.

if mel_knots_defined :
    # knotVals = mel_knots
    # use mel_knots for melodic contour spline
    # print("using mel_knots")
    print("")
else :
    mel_knots = getKnots(n_mel)

print("mel_knots: ")
print(mel_knots)

# finished config section

if stat == 1 :
# get the melodic contour spline values over [0,1] at 1000 points
# search through those 1000 points for stationary points, and also include endpoints
# getSplineVals returns [xvals, yvals] where xvals = [0, *, ..., *, 1] so this includes
# both endpoints and stationary points.
    mel_splineVals = getSplineVals(mel_bcoeffs, mel_knots, 1000)
    mel_Pts = getStatPts(mel_splineVals)
    notes = len(mel_Pts) - 1  # since note[i] will have duration x[i+1]-x[i]
    # notes is just the number of subintervals given by the list xvals 
else :
    if notes > 0 :
    # get spline y values to use for f0 of tones in melody 
        mel_splineVals = getSplineVals(mel_bcoeffs, mel_knots, notes)
        mel_Pts = []
        for i in range(notes + 1) :
            mel_Pts.append([mel_splineVals[0][i], mel_splineVals[1][i]])
    else :
        print("error: stat and notes are both zero")
        print("set stat = 1 to use stationary points, or set notes > 0 for regular sampling")
        
print("")
print("mel_Pts: ")
print(mel_Pts)
print("")
print("notes = ", notes)

# indices of key cycles are in keys from config
# these are floats, which is useful for scaling when f0 or time0 changes
# need to reset keys for f0 and time0
f0_scalar = f0 / 110.0

keys_float = keys   # placeholder keys for f0 = 110, time = 1 sec
temp = torch.tensor(keys_float)
temp *= f0_scalar
# print("temp: ", temp)
# keys_int is integer version of keys_float
keys_int = []
for i in range(num_keys) :
    keys_int.append(int(temp[i]))   
# print("keys_int: ", keys_int)
keys = keys_int
# keys are now set for first note with initial f0 and time = 1 sec
# other keys will be needed for each note dependent on new f0 and new time0
# so we need scale by 2^y (for f0) and by time0 for each note

note_keys = torch.zeros(notes, num_keys)

# old comments:
# mel_Pts is a list of pts [x,y] so we can use the 2^y in the usual way for 
# pitches and then get durations from x[i] - x[i-1], i=1 ... num_Pts - 1. 
# Those durations can then be scaled to make the first duration a chosen target value.
# Pitches are determined by single values, durations are determined as subintervals.

time1 = time0

# start_time will be advanced by note_length after each note is written to buffer
start_time = 0.0
# note_lengths are lengths of subintervals which represent note durations, from onset
# of given note, to onset of next note.  When using multiple voices, duration can be extended.
note_lengths = torch.zeros(notes)
# frequencies of notes:
frequencies = torch.zeros(notes)
# using gains on the key cycles is one way of imposing an envelope for each note
# so that linear interpolation between key cycles naturally produces an envelope.

x = 0.0
y = 0.0
t = 0.0
time2 = 0

# as we build the melody as a waveform, we need to set note_lengths (durations), 
# keys (indices of key cycles), and f0 (fundamental frequencies) according to the config params
# and spline values.  Note i starts at time x[i] with pitch f_0 * 2^y[i] and goes through time x[i+1]

# next loop goes from 0 to notes-1 and computes durations as subinterval lengths on x axis
# and removes pts with x values too close together, here less than 0.02 (2% on interval [0,1])
# running forwards (if retro == 0) or backwards (if retro == 1)
# but the pitch should still be computed at the left endpoint of subinterval

if stat == 1 :
    too_close = 1
    while too_close == 1 :
        too_close = 0
        notes = len(mel_Pts) - 1
        # print("notes = ", notes)
        for i in range(notes) :
            x0 = mel_Pts[i][0]
            x1 = mel_Pts[i+1][0]
            if x1 - x0 < 0.02 :
                # print("too close:  ", i, ": ", x0, i+1, ": ", x1, "diff:  ", x1-x0)
                too_close = 1
                to_remove = i+1
        if too_close == 1 :    
            # print("removing stat point: ", to_remove)
            del mel_Pts[to_remove]

print("")
print("mel_Pts after removals of close values:")
print(mel_Pts)
print("")

# mel_Pts are now set with xvals in [0,1] and yvals in [-1,1] so we can compute the first subinterval
# length: length0 = x1 - x0, and the duration scalar = time0 / length0 which will then scale the size
# of each note duration so that the first note has duration time0 as specified in config.

x0 = mel_Pts[0][0]
x1 = mel_Pts[1][0]
length0 = x1 - x0
duration_scalar = time0 / length0

for i in range(notes) :
    print("computing values for i = ", i)
    x0 = mel_Pts[i][0]
    x1 = mel_Pts[i+1][0]
    y = mel_Pts[i][1]
    if retro == 1 :
        x0 = mel_Pts[notes-i-1][0]
        x1 = mel_Pts[notes-i][0]
        y = mel_Pts[notes-i-1][1]
        print("x0: ", x0, "x1: ", x1)
    if invert == 1 :
        y *= -1
        print("y value is inverted: ", y)
    y *= scale
    print("y value is scaled by ", scale, ":  ", y)
    y += shift
    print("y value is shifted by ", shift, ":  ", y)
    y2 = np.exp2(y)
    print("y2 (exp2(y)) value: ", y2)
    note_lengths[i] = (x1 - x0) * duration_scalar
    print("note duration (scaled): ", note_lengths[i])
    frequencies[i] = f0 * y2
    print("frequencies[i]: ", frequencies[i])
    keys_float = keys   # placeholder keys for f0 from config, time = 1 sec
    temp = torch.tensor(keys_float)
    temp = y2 * temp  
    temp = note_lengths[i] * temp
    if voices > 1 :
        temp = voice_scalar * temp
    keys_int = []
    for j in range(num_keys) :
        keys_int.append(int(temp[j]))   
    note_keys[i] = torch.tensor(keys_int)
    print("scaled keys for i = ", i, keys_int)
    print("")

# above loop remains same as in mel5 without extending note values in voices (done below), 
# except if voices > 1 then keys are scaled by voice_scalar.

# note_lengths are now in seconds, scaled by duration_scalar to achieve time0 in config for first note.
# note_lengths[i] gives the length of time in which note i is the primary note sounding in the melody
# so this is the time duration from onset of note i to onset of note i+1.  When we have multiple voices
# then notes are written to buffers and are allowed to last for longer times to overlap.  This means
# we also need to extend the total melody duration to allow for some time at the end in which one or
# several notes are still sounding.  This extended time can be computed as the max value of the ending
# time of any extended note, which is the onset time plus the extended note time.

# loop through notes and compute onset time + extended note time = endtime

# the amount to scale note durations by for overlapping voices is set in config to voice_scalar

# set total time for melody to duration_scalar (this is determined by the duration for first note)
time2 = duration_scalar
print("total time: ", time2)
print("")

start_time = 0.0
note_endtime = 0.0
max_endtime = 0.0
# compute max_endtime
for i in range(notes) :
    note_endtime = start_time + note_lengths[i] * voice_scalar
    if note_endtime > max_endtime :
        max_endtime = note_endtime
    start_time += note_lengths[i] 

start_time = 0.0

# time2 = duration of entire waveform or melody in seconds
# each note is rounded up by one sample
waveform_length = int(time2 * sample_rate) + notes # in samples
if voices > 1 :
    waveform_length = int(max_endtime * sample_rate) + notes # in samples
    waveform = torch.zeros(voices, waveform_length)
else :
    waveform = torch.zeros(waveform_length)
    
print("voices = ", voices)
print("melody length in seconds: ", max_endtime)
print("melody length in samples: ", waveform_length)
print("waveform size: ", waveform.size())

x = 0.0
y = 0.0
t = 0.0
f = f0

index = audio_source.find(".")
audio_prefix = audio_source[:index]
print("audio_prefix:", audio_prefix)

# next loop writes each note in sequence to waveform and also
# prints ratios and cent values to compare notes in melody

cent_values = ""
f0_values = str(int(f0)) + " "
scale_values = "0  "
voice = 0

if retro == 1 :
    ratio = frequencies[0] / f0 
    centval = (1200 / np.log(2)) * np.log(ratio)
    scale_values = str(int(centval)) + "  "

for i in range(notes) :
    print("inserting for i = ", i)
    time1 = note_lengths[i]
    time_new = time1 * voice_scalar  # used below only if voices > 1
    f = frequencies[i]
    voice = i % voices
    print("into voice: ", voice) 
    print("start_time, time1, f0:  ", start_time, time1, f)
    if i > 0 :
        ratio = frequencies[i] / frequencies[i-1]
        centval = (1200 / np.log(2)) * np.log(ratio)
        print("interval:  ratio = ", ratio, " cent value =  ", centval)
        cent_values += str(int(centval)) + "  "
        ratio = frequencies[i] / f0 
        centval = (1200 / np.log(2)) * np.log(ratio)
        scale_values += str(int(centval)) + "  "
        f0_values += str(int(frequencies[i])) + "  "
        print("cent value relative to initial f0:  ", centval)
    print("note_keys[i]: ", note_keys[i])
#   print("key_gains: ", key_gains)
#   print("key_bcoeffs: ", key_bcoeffs)
#   print("key_knots: ", key_knots)
    print("interp_method: ", interp_method)
    if voices == 1 :
        insertWavTone2(waveform, start_time, f, time1, sample_rate, key_bcoeffs, key_knots, note_keys[i], key_gains, interp_method)
    if voices > 1 :
        insertWavTone3(waveform, voice, start_time, f, time_new, sample_rate, key_bcoeffs, key_knots, note_keys[i], key_gains, interp_method)
        # here we have changed time1 to time_new to reflect the new duration for overlaps of voices

    start_time += time1 * sample_rate
    print("")

if voices == 1 :
    waveform_out = torch.unsqueeze(waveform, dim=0)

# mix voices together:
if voices > 1 :
    mixed_waveform = torch.zeros(waveform_length)
    for j in range(waveform_length) :
        temp = 0.0
        count = 0.0
        for voice in range(voices) :
            if abs(waveform[voice, j]) > 0 :
                count += 1
                temp += waveform[voice, j] 
        mixed_waveform[j] = temp / voices
        # if count > 0 :
        #   mixed_waveform[j] = temp / count
    waveform_out = torch.unsqueeze(mixed_waveform, dim=0)

print("we have wav data")
print("now writing wav file")

if audio_source == "" :
    path = "../audio/melody-" + transform + ".wav"
    print("no audio source given, so output path is: ", path) 
else :
    index = audio_source.find(".")
    audio_prefix = audio_source[:index]
    path = audio_prefix + "/melody-" + transform + ".wav"
    if voices > 1 :
        path = audio_prefix + "/melody-" + transform + "-" + str(voices) + "voices" + ".wav"
    print("path for audio output: ", path)

torchaudio.save(
    path, waveform_out, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)

file_waveform, sample_rate = torchaudio.load(path)
np_waveform = file_waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("audio file has ", num_frames, " samples, at rate ", sample_rate)


file = transform + "-melody-summary.txt"
path = audio_prefix + "/" + file

summary = "melody summary:"
summary += "\n\n"
summary += "number of notes =  " + str(notes)
summary += ",  initial f0 = " + str(f0) + "\n\n"
summary += "first note duration =  " + str(time0)
summary += ",  total time in seconds = " + str(time2) + "\n\n"
summary += transform + " sequence of intervals between notes as cent values: \n"
summary += cent_values + "\n\n"
summary += transform + " sequence of intervals relative to initial f0 as 0: \n"
summary += scale_values + "\n\n"
summary += transform + " sequence of fundamental frequency f0 values: \n"
summary += f0_values + "\n\n"


with open(path, 'w') as f:
    f.writelines(summary)
    f.close()

print("")
print("")
print("********************************************************************")
print("")
print("")

print("copy of config file mel6config.txt :\n\n")

for i in range(len(config_lines)) :
    print(config_lines[i].rstrip())

print("")
print("")
print("********************************************************************")
print("")
print("")

print(summary)
