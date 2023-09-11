# this program records two seconds of audio and saves as output.wav and then reopens
# this file and does several spectrograms and predicts a sequence of f_0 for each segment.

# Sample rate is 16K, and CHUNK size for reading and also FFT size are both 1024.
# Hop size for FFT is 256, and our new segment size is 2048, which is about 1/8 sec.
# We read 32768 = 32 * 1024 samples, which can be partitioned into 16 segments.
# Each segment can hold (2048-1024)/256+1 = 5 FFT's, for example the first segment holds
# FFT's starting with indices 0, 256, 512, 768, and 1024, which have indices:
# [0,...,1023], [256,...,256+1023], [512,...,512+1023], [768,...,768+1023], and [1024,...,2047].
# We average these and take arg_max for our "weak" f_0 in each segment.

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

# these two functions compute arg_max on segments of audio, the first one also plots spec
# see bottom of this program to use one or the other in the code.
from argMaxSpec import plotSpecArgMax, getArgMax

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 44100
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
i = 0
# for RATE 16000 and CHUNK 1024 we get 2 * RATE/CHUNK = 31.125, so int() = 31
# so below we would read 0 to 31, or 32 chunks, so 32*1024 = 32768 samples
# or just over two seconds
for i in range(0, int(RATE / CHUNK * seconds)+1):
    data = stream.read(CHUNK)
    frames.append(data)
print("number of chunks: ", i+1)
    
print("recording stopped")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("output.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# end recording and writing wav file from mic
# begin PyTorch reading of wav file and do spectrogram

waveform, sample_rate = torchaudio.load("output.wav")
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape

segments = torch.tensor_split(waveform, 16, dim=1)

N = 1024
hop_size = 256

for seg_num in range(16) :
    
    waveform = segments[seg_num]
# to plot spec for each segment uncomment this next line (and fix spacing for python)
    arg_max = plotSpecArgMax(waveform, RATE, N, hop_size)
# to NOT plot spec for each segment use only this next line 
 #   arg_max = getArgMax(waveform, RATE, N, hop_size)
    samples_per_cycle = 16000 / arg_max
    left_endpoint = seg_num * float(1/8)
    right_endpoint = left_endpoint + float(1/8)
    output = str(seg_num).ljust(4,' ') + " arg_max in Hz: "
    output += str(arg_max).ljust(9,' ')
    output += " samples/cycle = "
    output += str(int(samples_per_cycle)).ljust(6,' ')
    output += " time interval: ["
    output += str(left_endpoint).ljust(6,' ')
    output += ","
    output += str(right_endpoint).ljust(6,' ')
    output += "]"
    print(output)
# end loop on segments

