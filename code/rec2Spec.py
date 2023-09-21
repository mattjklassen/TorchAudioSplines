# ----- Brief Description -----
# 
# Record two seconds of audio and save as output.wav, then reopen this file 
# and do several spectrograms and predict a sequence of f_0 for each segment.
# Output this info to console, or also to spectrogram graphs with matplot.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Defaults: sample rate is 16K, and CHUNK size for reading and also FFT size are both 1024.
# Hop size for FFT is 256, and segment size is 2048, which is about 1/8 sec.
# If we read 32768 = 32 * 1024 samples, this can be partitioned evenly into 16 segments.
# Each segment can hold (2048-1024)/256+1 = 5 FFT's, for example the first segment holds
# FFT's starting with indices 0, 256, 512, 768, and 1024, which have indices:
# [0,...,1023], [256,...,256+1023], [512,...,512+1023], [768,...,768+1023], and [1024,...,2047].
# We average these and take arg_max for our "weak" f_0 in each segment.
#
# ----- ----- ----- ----- -----


import pyaudio
import wave

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import os
import shutil
import requests

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=4)

# these two functions compute arg_max on segments of audio, the first one also plots spec
# see bottom of this program to use one or the other in the code.
from argMaxSpec import plotSpecArgMax, getArgMax

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
		channels=CHANNELS,
		rate=RATE,
		input=True,
		frames_per_buffer=CHUNK)

print("start recording ...")

frames = []
seconds = 2
for i in range(0, int(RATE / CHUNK * seconds)+1):
    data = stream.read(CHUNK)
    frames.append(data)

print("recording stopped")

print("size of chunk: ", CHUNK)
print("number of chunks in audio file: ", i+1)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("output.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# end recording from mic and writing wav file
# begin PyTorch reading of wav file and do spectrogram

waveform, sample_rate = torchaudio.load("output.wav")
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape

num_segments = 16
segments = torch.tensor_split(waveform, num_segments, dim=1)
print("number of segments: " , num_segments)
print("size of segments:")
for i in range(num_segments) :
    print("size of segment ", i, ":  ",  segments[i].size())

N = 1024
hop_size = 256

head = "segment"
head += "  "
head += "arg_max Hz"
head += "  "
head += "samples/cycle"
head += "  "
head += "time interval (secs)"
print(head)

for seg_num in range(16) :
    waveform = segments[seg_num]
    # to plot spec for each segment use next line
    # arg_max = plotSpecArgMax(waveform, RATE, N, hop_size)
    # to NOT plot spec for each segment use only next line 
    arg_max = getArgMax(waveform, RATE, N, hop_size)
    arg_max_str = f'{arg_max:.2f}'
    samples_per_cycle = 16000 / arg_max
    samples_per_cycle_str = f'{samples_per_cycle:.2f}'
    left_endpoint = seg_num * float(1/8)
    right_endpoint = left_endpoint + float(1/8)
    output = "   "
    output += str(seg_num).rjust(4,' ') 
    output += "      "
    output += str(arg_max_str).rjust(6,' ')
    output += "      "
    # output += str(samples_per_cycle_str).ljust(8)
    output += str(samples_per_cycle_str).rjust(6)
    output += "       ["
    output += str(left_endpoint).ljust(6,' ')
    output += ","
    output += str(right_endpoint).ljust(6,' ')
    output += "]"
    print(output)
# end loop on segments

