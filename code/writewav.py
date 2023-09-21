# ----- Brief Description -----
# 
# read a wav file at sample_rate (like 16K), do linear interpolation bewteen samples
# and write output at 3 * sample_rate (like 48K), wav file.  
# Assume both files' data is 16-bit, short ints.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# ----- ----- ----- ----- -----

# This script has no command line args

import io
import os
import numpy as np
import torch
import torchaudio

def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
    print()

print("reading audio file input.wav")
waveform, sample_rate = torchaudio.load("../audio/input.wav")
print("waveform.shape: ", waveform.shape)
# print("waveform.size(dim=0): ", waveform.size(dim=0))
# print("waveform.size(dim=1): ", waveform.size(dim=1))
# print("waveform[0,i] for i=0 to 9:")
# for i in range(10) :
#    print(waveform[0,i])
# waveform is a column vector of data in channel 0, for mono
# for stereo it would just be two columns for channels 0 and 1

np_waveform = waveform.numpy()
print("np_waveform.shape: ", np_waveform.shape)

num_channels, num_frames = np_waveform.shape
size_in = num_frames
data_in = torch.squeeze(waveform).numpy()
print("data_in = torch.squeeze(waveform).numpy()")
print("data_in.shape: ", data_in.shape)
# i^th sample value is now data_in[i]

size_out = 3 * size_in
data_out = torch.zeros(size_out)
print("data_out =  data_out = torch.zeros(size_out)")
print("data_out.shape: ", data_out.shape)
print("going back from data_out to waveform ...")

new_waveform = torch.empty(1, size_out)
print("new_waveform.shape: ", new_waveform.shape)
# new_waveform can hold one column of data (mono) with length size_out

# here we do the up-sampling with linear interpolation
for i in range(size_in) :
    new_waveform[0, 3 * i] = float(data_in[i])
    y1 = float(data_in[i])  
    y2 = y1
    if i+1 < size_in :
        y2 = float(data_in[i+1])
    # do linear interpolation between points (0,y1) and (1,y2) 
    # with (1-t)*y1+t*y2, for t=1/3 and t=2/3
    interp1 = (2/3)*y1 + (1/3)*y2
    interp2 = (1/3)*y1 + (2/3)*y2
    new_waveform[0, 3 * i + 1] = interp1 
    new_waveform[0, 3 * i + 2] = interp2
 
print("writing file newoutput.wav")
path = "../audio/newoutput.wav"
torchaudio.save(
    path, new_waveform, 3 * sample_rate,
    encoding="PCM_S", bits_per_sample=16)
inspect_file(path)

# sample terminal output:

# reading audio file input.wav
# waveform.shape:  torch.Size([1, 32768])
# np_waveform.shape:  (1, 32768)
# data_in = torch.squeeze(waveform).numpy()
# data_in.shape:  (32768,)
# data_out =  data_out = torch.zeros(size_out)
# data_out.shape:  torch.Size([98304])
# going back from data_out to waveform ...
# new_waveform.shape:  torch.Size([1, 98304])
# writing file newoutput.wav
# ----------
# Source: newoutput.wav
# ----------
# - File size: 196652 bytes
# - AudioMetaData(sample_rate=48000, num_frames=98304, num_channels=1, bits_per_sample=16, encoding=PCM_S)

