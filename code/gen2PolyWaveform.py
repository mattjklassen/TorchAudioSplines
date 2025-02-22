# ----- Brief Description -----
# 
# Generate waveform of one polyphonic cycle and model with bcoeffs.
# Bcoeffs file is first command line parameter.
# Frequencies can be input on command line as multipliers of 50 Hz. 
# So 4 5 6 7 would combine frequencies of 200, 250, 300, and 350 Hz.
# 
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Also create new folder inside polytones with name like 4-5-6-7.
# Put in here the bcoeffs used to form the mono cycles and also the
# bcoeffs for the polyphonic cycle. Also put graphs of those cycles.
# Then put wav file for polytone 2 sec, and any other melodic fragments
# generated with this polytone, including blending with other tones.
# For melodic fragments generated with mel.py also put the text summary.
#
# ----- ----- ----- ----- -----

import torch
import torchaudio
import numpy as np
import math
import sys

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 
from genCycle import genCycle, insertCycle, insertCycle2, insertCycle3
from getBcoeffs import getBcoeffs, import_bcoeffs, export_bcoeffs
from genWavTone import reset
import pathlib

# run example:
# python gen2PolyWaveform.py frhorn315/bcoeffs/bcoeffs-n30-seg0-cyc12.txt 4 5 6 7 3
# 4 5 6 7 are frequency multipliers, 3 is mult to scale the dimension for polycycle

bcoeffs_file = sys.argv[1]
bcoeffs = import_bcoeffs(bcoeffs_file)

print("imported bcoeffs:")
print(bcoeffs)

n = bcoeffs.size(dim=0)
f01 = float(200.0)
f02 = float(250.0)
f03 = float(300.0)
f03 = float(350.0)
f0 = float(50.0)
time = float(2.0)
# 2 second waveform
sample_rate = float(44100.0)

print("Argument List:", str(sys.argv))
args = len(sys.argv)
print("There are ", args, " args")

if args > 2 :
    s01 = sys.argv[2]
    int01 = int(s01)
    f01 = 50.0 * float(sys.argv[2])

if args > 3 :
    s02 = sys.argv[3]
    int02 = int(s02)
    f02 = 50.0 * float(sys.argv[3])

if args > 4 :
    s03 = sys.argv[4]
    int03 = int(s03)
    f03 = 50.0 * float(sys.argv[4])

if args > 5 :
    s04 = sys.argv[5]
    int04 = int(s04)
    f04 = 50.0 * float(sys.argv[5])

# mult is the multiplier to use for the dimension of the polyphonic cycle splines
# which is needed to give more accuracy to that spline over a longer interval.
# for example the polyphonic cycle might be 1/50 of a second and be mixed from
# cycles of length 1/200, 1/250, 1/300, and 1/350 (for the 4-5-6-7 case).
# so if the source bcoeffs have dimension say 30 (out of 220 samples per cycle for
# the f0=200 case) then with mult = 3 we have dimension 90 for the polyphonic cycle
# which has 882 samples per cycle.  30/200 = 0.15, 90/882 = 0.102.
if args > 6 :
    s05 = sys.argv[6]
    mult = int(s05)

print("frequences:")
print("f01: ", f01)
print("f02: ", f02)
print("f03: ", f03)
print("f04: ", f04)

path1 = "poly_tones/" 
path1 += s01 
path1 += "-" + s02 
path1 += "-" + s03 
path1 += "-" + s04 

pathlib.Path(path1).mkdir(parents=True, exist_ok=True)

# generate basic waveform for tone of length time in seconds:

def gen0WavTone(f0, time, sample_rate, bcoeffs) :

    cycle_length = float(1.0 / f0) # cycle length in seconds
    cycle_samples = int(sample_rate * cycle_length)
    poly_cycle = torch.zeros(cycle_samples) 
    num_cycles = int(f0 * time)
    waveform = torch.zeros(int(sample_rate * time)) 

    # first write poly_cycle 
    a = 0.0
    b = a + sample_rate * cycle_length
    b = reset(b)
    cycle = [a, b]
    insertCycle(poly_cycle, cycle, bcoeffs)

    for j in range(num_cycles) :
        for i in range(cycle_samples) :
            waveform[cycle_samples * j + i] = poly_cycle[i]

    return waveform


# generate polyphonic cycle with four fundamental frequencies using only one set of bcoeffs

def gen2PolyWaveform(f01, f02, f03, f04, f0, time, sample_rate, bcoeffs) :

    poly_cycle_length = float(1.0/50) # in seconds
    poly_cycle_samples = int(sample_rate * poly_cycle_length)
    poly_cycle = torch.zeros(poly_cycle_samples) 
    temp1 = torch.zeros(poly_cycle_samples) 
    temp2 = torch.zeros(poly_cycle_samples) 
    temp3 = torch.zeros(poly_cycle_samples) 
    temp4 = torch.zeros(poly_cycle_samples) 

    # first write 4 f01 cycles into poly_cycle
    num_cycles = int01
    cycle_length = sample_rate / f01 # cycle length in samples

    a = 0.0
    b = 0.0
    # write cycles 
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        insertCycle(temp1, cycle, bcoeffs)

    # next write 5 f02 cycles into poly_cycle
    num_cycles = int02
    cycle_length = sample_rate / f02 # cycle length in samples

    a = 0.0
    b = 0.0
    # write cycles 
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        insertCycle(temp2, cycle, bcoeffs)

    # next write 6 f03 cycles into poly_cycle
    num_cycles = int03
    cycle_length = sample_rate / f03 # cycle length in samples

    a = 0.0
    b = 0.0
    # write cycles 
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        insertCycle(temp3, cycle, bcoeffs)

    # next write 7 f04 cycles into poly_cycle
    num_cycles = int04
    cycle_length = sample_rate / f04 # cycle length in samples

    a = 0.0
    b = 0.0
    # write cycles 
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        insertCycle(temp4, cycle, bcoeffs)

    # mix the four pitches into one cycle: 
    for i in range(poly_cycle_samples) :
        poly_cycle[i] = (temp1[i] + temp2[i] + temp3[i] + temp4[i]) / 1
        # change to / 1 for a low gain waveform, can switch back to 4 for the strict average 

    # print("poly_cycle:")
    # print(poly_cycle)

    waveform_length = int(sample_rate * time)
    waveform = torch.zeros(waveform_length) 

    num_cycles = int(sample_rate * time / poly_cycle_samples)
    cycle_length = poly_cycle_samples

    for j in range(num_cycles) :
        for i in range(poly_cycle_samples) :
            waveform[poly_cycle_samples * j + i] = poly_cycle[i]

    return waveform

wav_data = gen2PolyWaveform(f01, f02, f03, f04, f0, time, sample_rate, bcoeffs)
print("we have wav data")
# wav_data is 2 seconds of harmonic seventh chord mixed with frequencies given (or other 4-note chord)
# ie. this is a mix of four buffers sample by sample, not a new spline approximation.

# write wav_data to file:
size_out = int(sample_rate * time)
waveform = torch.empty(1, size_out)
for i in range(size_out) :
    waveform[0,i] = wav_data[i]

path2 = path1 + "/poly_mix" 
numstr =  "-" + s01 +  "-" + s02 + "-" + s03 + "-" + s04 
path2 += "-" + s01 
path2 += "-" + s02 
path2 += "-" + s03 
path2 += "-" + s04 
path2 += ".wav"
print("now writing wav file:")
print(path2)

torchaudio.save(
    path2, waveform, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)

# now get new B-spline model coeffs of wav_data and write these bcoeffs to file:
poly_cycle_length = float(1.0/50) # in seconds
poly_cycle_samples = int(sample_rate * poly_cycle_length)
a = 0
b = poly_cycle_samples 
b = reset(b)
cycle = [a, b]
# n = bcoeffs.size(dim=0)
print("calling getBcoeffs")
poly_bcoeffs = getBcoeffs(waveform, sample_rate, cycle, mult * n)
print("finished getBcoeffs")

bcoeffs = []
for i in range(mult * n) :
    bcoeffs.append(float(poly_bcoeffs[i]))

file = path1 + "/poly_bcoeffs.txt"
export_bcoeffs(file, bcoeffs)
print("wrote bcoeffs to file")
# print("poly_bcoeffs:")
# print(bcoeffs)

new_bcoeffs = import_bcoeffs(file)
# print("new_bcoeffs:")
# print(new_bcoeffs)

# next write new file based on poly_bcoeffs 

print("calling gen0WavTone()")
wav_data = gen0WavTone(f0, time, sample_rate, new_bcoeffs)
print("finished gen0WavTone()")

size_out = int(sample_rate * time)
waveform = torch.empty(1, size_out)
for i in range(size_out) :
    waveform[0,i] = wav_data[i]

path3 = path1 + "/poly_bcoeffs" 
path3 += "-" + s01 
path3 += "-" + s02 
path3 += "-" + s03 
path3 += "-" + s04 
path3 += ".wav"
print("now writing wav file:")
print(path3)

torchaudio.save(
    path3, waveform, int(sample_rate),
    encoding="PCM_S", bits_per_sample=16)

cent_values = ""
print("")

summary = "polyphonic tone summary:"
summary += "\n\n"
summary += "bcoeffs source file:  " +  bcoeffs_file
summary += "\n\n"
summary += "ratios and cent values: \n\n"

ratio = float(int02 / int01)
centval = (1200 / np.log(2)) * np.log(ratio)
print("interval 1:  ratio = " + s02 + "/" + s01 + " with cent value =  ", centval)
summary += "interval 1:  ratio = " + s02 + "/" + s01 + " with cent value =  " + str(int(centval)) + "\n"
cent_values += str(int(centval)) + "  "

ratio = float(int03 / int02)
centval = (1200 / np.log(2)) * np.log(ratio)
print("interval 2:  ratio = " + s03 + "/" + s02 + " with cent value =  ", centval)
summary += "interval 2:  ratio = " + s03 + "/" + s02 + " with cent value =  " + str(int(centval)) + "\n"
cent_values += str(int(centval)) + "  "

ratio = float(int04 / int03)
centval = (1200 / np.log(2)) * np.log(ratio)
print("interval 3:  ratio = " + s04 + "/" + s03 + " with cent value =  ", centval)
summary += "interval 3:  ratio = " + s04 + "/" + s03 + " with cent value =  " + str(int(centval)) + "\n"
cent_values += str(int(centval)) + "  "
print("")

ratio = float(int04 / int01)
centval = (1200 / np.log(2)) * np.log(ratio)
print("interval 4:  ratio = " + s04 + "/" + s01 + " with cent value =  ", centval)
summary += "interval 4:  ratio = " + s04 + "/" + s01 + " with cent value =  " + str(int(centval))
cent_values += str(int(centval)) + "  "
print("")

# summary += cent_values

# retrieve and check
# waveform, sample_rate = torchaudio.load(path)
# np_waveform = waveform.numpy()
# num_channels, num_frames = np_waveform.shape
# length = num_frames / sample_rate
# print("input audio file has ", num_frames, " samples, at rate ", sample_rate)

# this is what genWavTone() is returning:
# waveform = torch.zeros(waveform_length)  # final output samples as 1-dim tensor array

path4 = path1 + "/summary" + numstr + ".txt"

with open(path4, 'w') as f:
    f.writelines(summary)
    f.close()



