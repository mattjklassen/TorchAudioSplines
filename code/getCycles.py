# ----- Brief Description -----
# 
# functions:  getCycles() and getf0withCycles()
# In the function getCycles we find "cycles" in an audio segment given weak f_0 (fundamental frequency).
# The weak f_0 is found by the function getArgMax() and the cycles are then found with getCycles()
# By "cycle" we mean a time interval [a,b] (with a and b time values in float samples)
# where time is measured from 0 in the audio segment, and where b-a has length in samples
# predicted by f_0, so b-a is approximately sample_rate * 1/f_0 (samples/cycle). 
# The function getf0withCycles() uses the above and then simply averages cycle lengths to get refined f0.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
#
# Cycles are computed based on zero crossings between samples, where the audio graph is computed
# using linear interpolation between samples.  Zero crossings are assumed to have the pattern of
# positive slope at t, so crossing of type f(x_i) < 0 and f(x_{i+1}) > 0 and x_i < t < x_{i+1}.
# Cycles of this type are found at each such zero crossing of positive slope, by finding the next
# such zero crossing which is closest to the project number of cycles per sample. Cycles may also overlap.
#
# We call f_0 "weak" since it can be applied in an audio segment which is not harmonic, meaning
# that it might not be deemed to have a fundamental frequency.  For example, a noisy segment may not
# have a clear f_0 but it can still be measured with STFT to produce weak f_0 which can simply be
# defined as an average arg_max of frequency bins.  Additionally, the values of the audio signal
# over the short interval of one cycle can still be used to represent the signal in the entire segment
# or an interval larger than one segment, using cycle interpolation.  Of course, the accuracy of such
# a representation between cycles is another matter.
#
# description of input parameters to function getCycles(waveform, rate, weakf0):
# 1. waveform -- is a torch tensor which should come from a line like:
# waveform, sample_rate = torchaudio.load("output.wav"), where output.wav is 2 sec at 16K sample rate
# shape of waveform is:  torch.Size([1, length]), where length is number of frames
# also np_waveform = waveform.numpy() and num_channels, num_frames = np_waveform.shape
# or could do segments = torch.tensor_split(waveform, 16, dim=1)
# then do waveform = segments[i], then for output.wav get segment size = 2000 samples
# 2. sample_rate -- audio sample rate
# 3. weakf0 -- predicted weak f_0
# it may also be interesting to have a measure of energy in the cycle, which can be computed
# as an average sum of squares of sample values, or spectral values
#
# ----- ----- ----- ----- -----

import torch
import torchaudio
import numpy as np


def getCycles(waveform, sample_rate, weakf0) :

    # This function searches through waveform to find intervals [a, b] called cycles.
    # Each cycle has the property that a and b are zero crossings of the piecewise linear
    # graph of waveform samples, each with positive slope, which means a and b satisfy:
    # a is a time value between some samples t_i and t_{i+1} with y_i < 0 < y_{i+1}
    # b is a time value between some samples t_j and t_{j+1} with y_j < 0 < y_{j+1}
    # and further we require that the distance b - a is chosen as close as possible to
    # the predicted cycle length which should be approximately (sample_rate / weakf0) samples. 
    # The function returns a list of the intervals as pairs a,b.  The intervals [a,b] may overlap. 

    # One more thing: It turns out to simplify things if we assume that zero crossings never
    # occur exactly at sample values.  Since we don't have any exact information about the waveform 
    # between samples it is not a stretch to move a zero crossing that occurs (almost) exactly at a
    # sample value to the right or left by say 0.01 samples.  This could also be re-adjusted if we
    # do a sample rate change.  Not sure if this will cause other problems yet. 

    a = 0.0  # left endpoint of cycle
    b = 0.0  # right endpoint of cycle
    np_waveform = waveform.numpy() 
    num_channels, num_frames = np_waveform.shape
    # weakf0 is cycles/sec, sample_rate is samples/sec 
    # sample_rate/weakf0 is samples/cycle = cycle length in samples
    weakT0 = float(sample_rate) / weakf0 # predicted cycle length in samples

    # print("shape of np_waveform  ", np_waveform.shape)
    y0 = 0.0
    y1 = 0.0
    zero = 0.0
    end_pts = []
    zeros = []
    cycles = []

    # loop over samples in waveform to find zeros with positive slope:
    for i in range(int(num_frames - 2)) :
        y0 = np_waveform[0,i]
        y1 = np_waveform[0,i+1]
        if (y0 < 0) and (y1 > 0) :  # positive slope and zero crossing conditions met
#            print("sample ", i, " : ", y0)
#            print("sample ", i+1, " : ", y1)
            m = y1 - y0  # line is y(t) = y0 + mt = 0 when t = -y0/m
            zero = float(i) - y0 / m
            end_pts.append([y0,y1])
            zeros.append(zero)
    # print("zeros:")
    # print(zeros)

    previous_closest = 0
    previous_diff = 100000
    previous_error = 100000 
    num_zeros = len(zeros)
    last_zero = zeros[num_zeros-1]
    for i in range(num_zeros-1) :
        exceeded = False
        temp = zeros[i] + weakT0  # search for next zero at period guess weakT0
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
        a = zeros[i]
        b = zeros[closest]
        diff = b - a
        # each cycle is a list [a, b]
        error = abs(diff - weakT0)
        num_cycles = len(cycles)
	# only append new cycle if it has a different b from previous, 
    # or its error is smaller, ie. length diff is closer to weakT0 than previous.
	# If this latter condition happens then delete previous cycle and append new.
        if closest == previous_closest :
            if error < previous_error :
                cycles.pop()
                cycles.append([a, b])
        else :
            cycles.append([a, b])

        previous_closest = closest
        previous_diff = diff
        previous_error = error

    return cycles


def getConsecutiveCycles(waveform, sample_rate, weakf0) :

    # Same as above but now force cycles to be consecutive, ie. [z_1,z_2] must be followed
    # by [z_2,z_3] where the z_i are zeros with positive slope.  There may be other zeros
    # with positive slope which are skipped and do not form the endpoints of cycles.
    # 
    # previous comments:
    # 
    # This function searches through waveform to find intervals [a, b] called cycles.
    # Each cycle has the property that a and b are zero crossings of the piecewise linear
    # graph of waveform samples, each with positive slope, which means a and b satisfy:
    # a is a time value between some samples t_i and t_{i+1} with y_i < 0 < y_{i+1}
    # b is a time value between some samples t_j and t_{j+1} with y_j < 0 < y_{j+1}
    # and further we require that the distance b - a is chosen as close as possible to
    # the predicted cycle length which should be approximately (sample_rate / weakf0) samples. 
    # The function returns a list of the intervals as pairs a,b.  The intervals [a,b] may overlap. 

    # One more thing: It turns out to simplify things if we assume that zero crossings never
    # occur exactly at sample values.  Since we don't have any exact information about the waveform 
    # between samples it is not a stretch to move a zero crossing that occurs (almost) exactly at a
    # sample value to the right or left by say 0.01 samples.  This could also be re-adjusted if we
    # do a sample rate change.  Not sure if this will cause other problems yet. 

    a = 0.0  # left endpoint of cycle
    b = 0.0  # right endpoint of cycle
    np_waveform = waveform.numpy() 
    num_channels, num_frames = np_waveform.shape
    # weakf0 is cycles/sec, sample_rate is samples/sec 
    # sample_rate/weakf0 is samples/cycle = cycle length in samples
    weakT0 = float(sample_rate) / weakf0 # predicted cycle length in samples

    # print("shape of np_waveform  ", np_waveform.shape)
    y0 = 0.0
    y1 = 0.0
    zero = 0.0
    end_pts = []
    zeros = []
    cycles = []

    # loop over samples in waveform to find zeros with positive slope:
    for i in range(int(num_frames - 2)) :
        y0 = np_waveform[0,i]
        y1 = np_waveform[0,i+1]
        if (y0 < 0) and (y1 > 0) :  # positive slope and zero crossing conditions met
        #    print("sample ", i, " : ", y0)
        #    print("sample ", i+1, " : ", y1)
            m = y1 - y0  # line is y(t) = y0 + mt = 0 when t = -y0/m
            zero = float(i) - y0 / m
            end_pts.append([y0,y1])
            zeros.append(zero)

    num_zeros = len(zeros)
    last_zero = zeros[num_zeros-1]

    print("number of zeros:  ", num_zeros)
    print("last zero:  ", last_zero)

    a = zeros[0]
    b = a

    exceeded = False
    counter = 0

    while (not exceeded) :
        a = b
        if (last_zero - a) < weakT0 :
            print("a is within weakT0 of last zero")
            exceeded = True
            break
        temp = b + weakT0  # search for next zero at period guess weakT0
        if temp > last_zero :
            print("temp exceeds last_zero")
            exceeded = True
            break
        j = 0
        while zeros[j] < temp :
            j += 1
            if j > num_zeros - 1 :
                j = num_zeros - 1
                break
        closest = j
        # if abs(zeros[j] - temp) > abs(zeros[j-1] - temp) :
        #     closest = j-1
        if abs(zeros[j] - a) < 0.001 :
            closest += 1
        # if exceeded :
        #    closest = num_zeros - 1
        # if closest == i :
        #     closest = i + 1
        # if closest > num_zeros - 1 :
        #     closest = num_zeros - 1
        b = zeros[closest]
        diff = b - a
        # each cycle is a list [a, b]
        error = abs(diff - weakT0)
        num_cycles = len(cycles)
        cycles.append([a, b])
        counter += 1
        print("counter: ", counter, "  a = ", a, "  b = ", b, "  diff = ", diff)
        print("last_zero - a :  ", last_zero - a)
        if counter > 700 :
            exit(0)

    return cycles


def getf0withCycles(waveform, sample_rate, weakf0) :

    cycles = getCycles(waveform, sample_rate, weakf0)

    avg_cycle_len = 0.0
    for i in range(len(cycles)) :
        cycle = cycles[i]
        a = cycle[0]
        b = cycle[1]
        cycle_len = b - a
        avg_cycle_len += cycle_len
    avg_cycle_len /= len(cycles)
    avg_f0 = sample_rate / avg_cycle_len

    return avg_f0


