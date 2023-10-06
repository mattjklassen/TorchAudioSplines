
# ----- Brief Description -----
# 
# In this script we use getf0withCycles() applied to an audio file input.
# (change audio file with path variable below ...)
# Briefly, this constructs an estimate of f0 by first doing STFT and argMax
# then refining this estimate using zero crossings to form cycles with getCycles()
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Cycles are computed with getCycles based on zero crossings with positive slope
# 
# 


import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from argMaxSpec import plotSpecArgMax, getArgMax
from cycleSpline import plotCycleSpline
from getCycles import getCycles, getf0withCycles


# testing getf0withCycles()

# path = "../audio/input.wav"
path = "../audio/A445.wav"
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)

# split waveform into segments, 
# example: sample rate = 16000, segments of length 2048 samples, num_segments = 16

num_segments = int(num_frames / 2048)
segments = torch.tensor_split(waveform, num_segments, dim=1)
segment_size = num_frames / num_segments
print("splitting into ", num_segments, " segments")
# for i in range(num_segments) :
#    print("size of segment ", i, " : ", segments[i].size())

RATE = sample_rate
N = 1024
# hop_size = 256
hop_size = 128
energy = 0.0

for i in range(num_segments) :

    current_segment = i
    print("")
    print("segment number ", current_segment)
    segment_start = segment_size * current_segment
    segment_end = segment_start + segment_size
    waveform = segments[current_segment]
    np_waveform = waveform.numpy() 
    data = torch.squeeze(waveform).numpy()
    # i^th sample value is now data[i]
    num_channels, num_frames = np_waveform.shape
    
    # get the weak f_0 (approx fundamental frequency) with getArgMax
    max_f0 = 800
    arg_max = getArgMax(waveform, RATE, N, hop_size, max_f0)
    arg_max_str = f'{arg_max:.2f}'
    samples_per_cycle_guess = RATE / arg_max
    spc_str = f'{samples_per_cycle_guess:.2f}'
    num_hops = int((num_frames - N)/hop_size)+1
    # print("arg_max:  ", arg_max_str)
    # print("samples per cycle guess:  ", spc_str)
    # print("num_frames: ", num_frames)
    # print("FFT size N: ", N)
    # print("hop_size: ", hop_size)
    # print("number of hops: ", num_hops)
    # print("(num_hops-1) * hop_size + N = ", (num_hops - 1) * hop_size + N)
    
    # get cycles according to predicted f_0
    cycles = getCycles(waveform, RATE, arg_max)
    num_cycles = len(cycles)
    f0 = getf0withCycles(waveform, RATE, arg_max)
    print("f0 with Cycles for segment: ", i, " : ", f0)




