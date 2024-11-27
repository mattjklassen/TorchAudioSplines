

# ----- Brief Description -----
# 
# Generate waveform of one polyphonic cycle and model with bcoeffs.
# This one does a harmonic seventh with f01 = 200, f02 = 250, f03 = 300, f04 = 350.
# 

# ----- ----- ----- ----- -----

# ------- More Details --------
# 
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
from getBcoeffs import import_bcoeffs, export_bcoeffs
from genWavTone import reset

bcoeffs_file = sys.argv[1]
bcoeffs = import_bcoeffs(bcoeffs_file)
f01 = float(200.0)
f02 = float(250.0)
f03 = float(300.0)
f04 = float(350.0)
f0 = float(50.0)
time = float(2.0)
# 2 second waveform
sample_rate = float(44100.0)


# generate polyphonic cycle with two fundamental frequencies f01 and f02 and time = length in seconds
# using only one set of bcoeffs, with typical case: f01=200, f02=300, time=1/100 sec, so that there are
# 2 cycles at f01 and 3 cycles at f02 mixed into polyphonic cycle.
# output: tensor of floats as output sample values

def gen2PolyWaveform(f01, f02, f03, f04, f0, time, sample_rate, bcoeffs) :

    poly_cycle_length = float(1.0/50) # in seconds
    poly_cycle_samples = int(sample_rate * poly_cycle_length)
    poly_cycle = torch.zeros(poly_cycle_samples) 
    temp1 = torch.zeros(poly_cycle_samples) 
    temp2 = torch.zeros(poly_cycle_samples) 
    temp3 = torch.zeros(poly_cycle_samples) 
    temp4 = torch.zeros(poly_cycle_samples) 

    # first write 4 f01 cycles into poly_cycle
    num_cycles = 4 
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
    num_cycles = 5 
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
    num_cycles = 6 
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
        insertCycle(temp2, cycle, bcoeffs)

    # next write 7 f04 cycles into poly_cycle
    num_cycles = 7 
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
        insertCycle(temp2, cycle, bcoeffs)

    for i in range(poly_cycle_samples) :
        poly_cycle[i] = (temp1[i] + temp2[i] + temp3[i] + temp4[i]) / 1

    print("poly_cycle:")
    print(poly_cycle)

    waveform_length = int(sample_rate * time)
    waveform = torch.zeros(waveform_length) 

    num_cycles = int(sample_rate * time / poly_cycle_samples)
    cycle_length = poly_cycle_samples

    for j in range(num_cycles) :
        for i in range(poly_cycle_samples) :
            waveform[poly_cycle_samples * j + i] = poly_cycle[i]

    a = 0.0
    b = 0.0
    # write cycles 
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
#        insertCycle(waveform, cycle, bcoeffs)

    return waveform

# wav_data is 2 seconds of harmonic seventh chord mixed with frequencies given
wav_data = gen2PolyWaveform(f01, f02, f03, f04, f0, time, sample_rate, bcoeffs)

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





