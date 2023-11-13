
# ----- Brief Description -----
# 
# create directory material<number> and put melodic segments with transformed versions
# and report into directory.  Report should include plot of splines used also.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# for example: material1/ could contain:
# x. plots of spline cycles 
# x. bcoeffs of spline cycles 
# x. wav files of tones and scales
# x. key cycle sequences of tones
# x. wav files of spline melodic fragments
# x. wav files of transformed melodic fragments
# 
# There are various transforms to use:
# x. trasposition (can be done in the DAW)
# x. inversion and retrograde for both pitch and time
# x. time stretching
#
# ----- ----- ----- ----- -----


import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 

from argMaxSpec import plotSpecArgMax, getArgMax
from cycleSpline import plotCycleSpline
from matplotlib.backends.backend_pdf import PdfPages
from getCycles import getCycles

print("Argument List:", str(sys.argv))

audiofile = sys.argv[1]
prefix = "../audio/"
n = int(sys.argv[2]) # dimension of C^2 cubic spline vector space V


# start_sample = int(sys.argv[2])
# end_sample = int(sys.argv[3])
# count = number of samples for graph, ends included
# count = end_sample - start_sample + 1 

d = 3     # degree for cubic splines
k = n - d # number of subintervals

audio_file = "A445.wav"
new_dir = "material" + audio_file + "dim" + str(n)
print("will create: ", new_dir)
# os.mkdir(new_dir) 

# The idea is to put into this directory a bunch of data and graphs and musical material.
# There can be scales, melodic fragments, harmonic fragments, pdf's of cycle graphs,
# B-spline coefficients for cycles, etc.

# First process audio file, then compute segments.  Then do loop on segments to find cycles,
# write bcoeff files, and write pdf graphs of cycles.  Could also limit the selection of cycles.  
# Then form scales, melodic fragments and other material.

path = prefix + audiofile
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate
segment_size = 2048
num_segments = int(num_frames / segment_size)
print("input audio file has ", num_frames, " samples, at rate ", sample_rate)

# split waveform into segments

segments = get_segments(waveform, sample_rate, segment_size)

N = 1024
hop_size = 128
# energy = 0.0

# loop on segments (get cycles, compute bcoeffs, graph cycles ...)

for i in range(num_segments) :

    segment = segments[i]
    current_segment = i
    process_segment(segment, current_segment, segment_size, sample_rate, N, hop_size)



