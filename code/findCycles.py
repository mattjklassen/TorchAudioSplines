# In this script we do three main things:
# 1. find weak f_0 with getArgMax()
# 2. find cycles with findCycles()
# 3. produce output to pdf

# In the function findCycles we find "cycles" in an audio segment given weak f_0 (fundamental frequency).
# The weak f_0 is found by the function getArgMax() and the cycles are then found with getCycles()
# By "cycle" we mean a time interval [a,b] (with a and b time values in float samples)
# where time is measured from 0 in the audio segment, and where b-a has length in samples
# predicted by f_0, so b-a is approximately sample_rate * 1/f_0 (samples/cycle). 

# Cycles are computed based on zero crossings between samples, where the audio graph is computed
# using linear interpolation between samples.  Zero crossings are assumed to have the pattern of
# positive slope at t, so crossing of type f(x_i) < 0 and f(x_{i+1}) > 0 and x_i < t < x_{i+1}.
# Cycles of this type are found at each such zero crossing of positive slope, by finding the next
# such zero crossing which is closest to the project number of cycles per sample. Cycles may also overlap.

# We call f_0 "weak" since it can be applied in an audio segment which is not harmonic, meaning
# that it might not be deemed to have a fundamental frequency.  For example, a noisy segment may not
# have a clear f_0 but it can still be measured with STFT to produce weak f_0 which can simply be
# defined as an average arg_max of frequency bins.  Additionally, the values of the audio signal
# over the short interval of one cycle can still be used to represent the signal in the entire segment
# or an interval larger than one segment, using cycle interpolation.  Of course, the accuracy of such
# a representation between cycles is another matter.

# pdf output is summarized below

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
# import fpdf

from argMaxSpec import plotSpecArgMax, getArgMax
from cycleSpline import plotCycleSpline
from matplotlib.backends.backend_pdf import PdfPages

# input parameters to getCycles(waveform, rate, freq):

# 1. waveform is a torch tensor which should come from a line like:
# waveform, sample_rate = torchaudio.load("output.wav"), where output.wav is 2 sec at 16K sample rate
# shape of waveform is:  torch.Size([1, length]), where length is number of frames
# also np_waveform = waveform.numpy() and num_channels, num_frames = np_waveform.shape
# or could do segments = torch.tensor_split(waveform, 16, dim=1)
# then do waveform = segments[i], then for output.wav get segment size = 2000 samples

# 2. rate = audio sample rate

# 3. freq = predicted weak f_0
# it may also be interesting to have a measure of energy in the cycle, which can be computed
# as an average sum of squares of sample values, or spectral values

# output is written to pdf with title page first, then plot of waveform, and plots of all cycles found

def getCycles(waveform, rate, freq) :

    # need to find time value a between samples t_i and t_{i+1} with y_i < 0 < y_{i+1}
    a = 0.0  # left endpoint of cycle
    b = 0.0  # right endpoint of cycle
    cycle_length = 0.0
    np_waveform = waveform.numpy() 
    num_channels, num_frames = np_waveform.shape
    # freq is cycles/sec, rate is samples/sec 
    # rate/freq is samples/cycle = cycle length in samples
    cycle_length = float(rate) / freq  # cycle length in samples

    # print("shape of np_waveform  ", np_waveform.shape)
    y0 = 0.0
    y1 = 0.0
    zero = 0.0
    end_pts = []
    zeros = []
    cycles = []

    # loop over samples in waveform to find a and b
    for i in range(int(num_frames - 2)) :
        y0 = np_waveform[0,i]
        y1 = np_waveform[0,i+1]
        if (y0 < 0) and (y1 > 0) :
#            print("sample ", i, " : ", y0)
#            print("sample ", i+1, " : ", y1)
            m = y1 - y0  # line is y(t) = y0 + mt = 0 when t = -y0/m
            zero = float(i) - y0 / m
            end_pts.append([y0,y1])
            zeros.append(zero)
    # print("zeros:")
    # print(zeros)

    num_zeros = len(zeros)
    last_zero = zeros[num_zeros-1]
    for i in range(num_zeros-1) :
        exceeded = False
        temp = zeros[i] + cycle_length
        if temp > last_zero :
            # print("temp exceeds last_zero")
            exceeded = True
        j = 0
        while zeros[j] < temp :
            j += 1
            if j > num_zeros - 1 :
                j = num_zeros - 1
                break
        closest = j
        if abs(zeros[j] - temp) > abs(zeros[j-1] - temp) :
            closest = j-1
        if exceeded :
            closest = num_zeros - 1
        if closest == i :
            closest = i + 1
        if closest > num_zeros - 1 :
            closest = num_zeros - 1
        diff = zeros[closest] - zeros[i]
        # each cycle is a list [a, b]
        cycles.append([zeros[i],zeros[closest]])

    return cycles

	    
#    print("end_pts array:")
#    print(end_pts)
#    print(end_pts[2])

# test the function getCycles with waveform data and write output to pdf
# start with waveform of about 2 seconds length, at sample rate 16000

path = "../audio/output.wav"
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
length = num_frames / sample_rate

#split waveform into 16 segments, each of length 2048 samples
num_segments = 16
segments = torch.tensor_split(waveform, num_segments, dim=1)
segment_size = num_frames / num_segments

RATE = 16000
N = 1024
hop_size = 256
energy = 0.0

# for seg_num in range(16) :
# waveform = segments[seg_num]

# assigning a particular segment for testing
current_segment = 5
segment_start = segment_size * current_segment
segment_end = segment_start + segment_size
waveform = segments[current_segment]
np_waveform = waveform.numpy() 
data = torch.squeeze(waveform).numpy()
# i^th sample value is now data[i]
num_channels, num_frames = np_waveform.shape

# get the weak f_0 (approx fundamental frequency) with getArgMax
arg_max = getArgMax(waveform, RATE, N, hop_size)
samples_per_cycle_guess = RATE / arg_max
print("arg_max:  ", arg_max)
print("samples per cycle guess:  ", RATE / arg_max)
print("num_frames: ", num_frames)

# get cycles according to predicted f_0
cycles = getCycles(waveform, RATE, arg_max)
num_cycles = len(cycles)

# print pdf with title page first, then plot of waveform, and plots of all cycles found:
pp = PdfPages('../doc/out.pdf')

firstPage = plt.figure(figsize=(15,8))
firstPage.clf()
txt1 = "Audio File read: " + path 
txt1 += "      Length in seconds: " + str(length) 
txt1 += "      Sample Rate:  " + str(RATE)
txt2 = "Number of Segments:  " + str(num_segments)
txt2 += "      Segment Size:  " + str(segment_size)
txt2 += "      FFT Size:  " + str(N)
txt2 += "      Hop Size:  " + str(hop_size)
txt3 = "Data for Segment " + str(current_segment) + ":"
txt3 += "     Weak f_0:  " + str(arg_max) + " Hz"
txt3 += "     Target Samples per Cycle:  " + str(round(samples_per_cycle_guess,1)).ljust(8, ' ')
txt3 += "    Number of Cycles:  " + str(num_cycles)
txt4 = "Cycle Number:" 
txt4 = str(txt4).ljust(25, ' ')
txt5 = "Samples per Cycle:"
txt5 = str(txt5).ljust(25, ' ')

firstPage.text(0.1,0.8,txt1, transform=firstPage.transFigure, size=14)
firstPage.text(0.1,0.75,txt2, transform=firstPage.transFigure, size=14)
firstPage.text(0.1,0.65,txt3, transform=firstPage.transFigure, size=14)

lines = int(num_cycles / 10)
remainder = num_cycles % 10
print("lines: ", lines)
print("remainder: ", remainder)
print("lines * 10 + remainder - num_cycles :  ", lines * 10 + remainder - num_cycles)

def printOneLine(line) :
    start = 0.23
    firstPage.text(0.1, 0.55 - line * 0.1, txt4, transform=firstPage.transFigure, size=14)
    firstPage.text(0.1, 0.5  - line * 0.1, txt5, transform=firstPage.transFigure, size=14)
    for i in range(10) :
        j = i + 10 * line
        a = cycles[j][0]
        b = cycles[j][1]
        samples = int(b-a)
        txt6 = str(j).rjust(8, ' ')
        txt7 = str(samples).rjust(8, ' ')
        firstPage.text(start + i * 0.05, 0.55 - line * 0.1, txt6, transform=firstPage.transFigure, size=14)
        firstPage.text(start + i * 0.05, 0.5  - line * 0.1, txt7, transform=firstPage.transFigure, size=14)

def printLastLine() :
    start = 0.23
    firstPage.text(0.1, 0.55 - lines * 0.1, txt4, transform=firstPage.transFigure, size=14)
    firstPage.text(0.1, 0.5  - lines * 0.1, txt5, transform=firstPage.transFigure, size=14)
    for i in range(remainder) :
        j = i + 10 * lines
        a = cycles[j][0]
        b = cycles[j][1]
        samples = int(b-a)
        txt6 = str(j).rjust(8, ' ')
        txt7 = str(samples).rjust(8, ' ')
        firstPage.text(start + i * 0.05, 0.55 - lines * 0.1, txt6, transform=firstPage.transFigure, size=14)
        firstPage.text(start + i * 0.05, 0.5  - lines * 0.1, txt7, transform=firstPage.transFigure, size=14)

for i in range(lines) :
    printOneLine(i)

printLastLine()

plt.savefig(pp, format='pdf')

# title page is finished
# next do waveform segment graph page

fig = plt.figure(figsize=(15,8))
times = np.linspace(0, num_frames, num=num_frames)
samples = np.zeros(num_frames)
for i in range(num_frames) :
    samples[i] = data[i]

plt.plot(times, samples)
segment_title = "segment " + str(current_segment) + "  : "
segment_title += str(segment_size) + " samples: (" + str(int(segment_start)) + " to " + str(int(segment_end)) + ")"
plt.title(segment_title)
plt.ylabel("sample float values")
plt.xlabel("time in samples")
plt.savefig(pp, format='pdf')

# waveform segment graph page is finished
# next do one page per cycle graph

for i in range(num_cycles) :
    a = cycles[i][0]
    b = cycles[i][1]
    print("cycle ", i, " a: ", a, " b: ", b)
    n = 30
    fig = plotCycleSpline(waveform, i, a, b, n)
    plt.savefig(pp, format='pdf')

pp.close()



