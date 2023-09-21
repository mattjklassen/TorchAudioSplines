# ----- Brief Description -----
# 
# This program takes a wav file and begin and end sample numbers as command line input
# and draws a plot of the wav file sample values with matplot as piecewise linear graph.
# The sample values are also printed out as text on the command line.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# There is no error checking, for example to see if begin and end sample numbers are within range.
#
# ----- ----- ----- ----- -----


import wave
import matplotlib.pyplot as plt
import numpy as np
import sys

print("This is the name of the program:", sys.argv[0])
print("Argument List:", str(sys.argv))

audiofile = sys.argv[1]
start_sample = int(sys.argv[2])
end_sample = int(sys.argv[3])
k = end_sample - start_sample + 1

obj = wave.open(audiofile, "rb")
sr = obj.getframerate()
start_time = start_sample / sr
end_time = end_sample / sr
length_in_sec = k / sr
nsamp = obj.getnframes()
signal = obj.readframes(-1)

obj.close()

signal_len = nsamp / sr

print("num samples:  ", nsamp)
print("sample rate:  ", sr)
print("signal length in sec:  ", signal_len)

data_int = np.frombuffer(signal, dtype=np.int16)
data = data_int.astype('float32')
data /= 32768.0
times = np.linspace(0, signal_len, num=nsamp)

# k = 50
steps = k + 1
short_times = np.linspace(start_time, end_time, num=steps)
short_data = np.zeros(steps)
for i in range(0, steps) :
    short_data[i] = data[start_sample + i]

plt.figure(figsize=(15,8))
# plt.plot(times, data)
plt.plot(short_times, short_data)
plt.title(audiofile + " segment: samples " + str(start_sample) + " to " + str(end_sample))
plt.ylabel("sample float values")
plt.xlabel("time in seconds")
# plt.xlim(0, signal_len)
plt.xlim(start_time, end_time)
plt.show()

print("some sample values:")
for i in range(start_sample, end_sample) :
    print("sample number ", i, data[i])

