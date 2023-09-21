# ----- Brief Description -----
# 
# Here we convert the script wavspline.py into a function with parameters:
# waveform, end_points a and b (float time values measured in samples), and n (spline dim).
# Output to console gives audio file and segment info,
# output with matplot has audio graph as piecewise linear in grey
# overlayed with spline curve in green and n interpolation points highlighted in red.
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Another important difference from wavspline.py is that we allow endpoints of interval [a,b]
# to be float time values measured in samples, instead of integer sample values only.
# Instead of reading wav file, we start with the array of sample data as tensor or numpy array.
# The graph of signal is piecewise linear, connecting sample values by linear interpolation
# starting with math.floor(a) on the left, and ending with math.ceil(b) on the right.
# The graph of spline covers only the smaller interval [a,b].
#
# (this paragraph repeated from wavspline.py)
# Important note: the spline interpolation points are not uniform. We did this when
# forming splines to model cycles by choosing first a uniform sequence of k subintervals
# for the cubic spline and a knot sequence for the basis of B-splines for the vector 
# space V of C^2 cubic splines on the intervals [0,1/k,2/k,...,(k-1)/k,1].  We also chose
# the knot sequence 0,0,0,0,1/k,2/k,...,(k-1)/k,1,1,1,1 since this gives a B-spline basis
# with the simple properties that for any f(t) given as a sum of the basis B-splines
# f(t) = SUM_{i=0}^{n-1} c_i B^3_i(t), where n=k+3 is the dimension of V, we have
# f(0) = c_0, f(1) = c_{n-1}, f'(0) = c_1, and f'(1) = c_{n-2}.  Further, the interpolation
# points (s_i,y_i) i=0,...,n-1 should be chosen on the interval [0,1] according to the
# Schoenberg-Whitney Theorem so that B^3_i(s_i) > 0, or equivalently such that 
# t_i < s_i < t_{i+4} for each i.  A uniform sequence s_i = i/(n-1) works for i=0,...,n-1
# but we chose to use the sequence of k+1 endpoints of subintervals and then to insert
# two more values to total n=k+3 at the midpoints of the first and last subintervals,
# at 1/(2k) and 1-1/(2k).  These values have the function of giving a slightly "tighter"
# fit of the spline at the endpoints.  Since we are fitting to audio data of one cycle
# at a time, this makes it more likely that there will be good agreement of derivatives 
# where cycle splines join at the endpoints.
#
# ----- ----- ----- ----- -----


import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import math

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 

def plotSegmentSpline(waveform, a, b, n) :

    # example: waveform has 2048 samples x_0 to x_2047 and [a,b] = [15.6,235.1]
    # typically this means that 15.6 and 235.1 are zeros of the piecewise linear audio graph
    # and that we are choosing an approximate cycle of length 235.1 - 15.6 = 219.5 which
    # has been predicted as weak f_0 for this segment of audio.
    # Then we should have start_sample = 15 and end_sample = 236. 
    
    # In order to compute the piecewise linear values from the audio graph on [a,b], 
    # we need to have all relevant samples, so start with int(a) which <= a, and end
    # end with -int(-b) which is >= b.  Then the samples from start_sample to end_sample
    # can be used to compute any piecewise linear value in [a,b].

    print("use floor and ceil of a and b: ")
    start_sample = math.floor(a)
    end_sample = math.ceil(b)
    print("a = ", a, "  b = ", b)
    print("start_sample = ", start_sample, "  end_sample = ", end_sample)


    n = 30
    # count is the number of samples used to compute all values in [a,b]
    count = int(end_sample - start_sample) + 1
    print("number of samples in interval [start_sample, end_sample]:  ", count)
    
    # assume waveform passed to this function comes from read of this type: 
    # waveform, sample_rate = torchaudio.load("output.wav")
    # np_waveform = waveform.numpy()
    # num_channels, num_frames = np_waveform.shape
    # segments = torch.tensor_split(waveform, 16, dim=1)
    # waveform = segments[0] 
    # (now waveform has 2000 samples if sample_rate = 16000 and audio file is 2 sec)
    # (or 2048 if file has length 32768 samples) 

    np_waveform = waveform.numpy() 
    num_channels, num_frames = np_waveform.shape
    # i^th sample value is now np_waveform[0,i] (but becomes data[i] below)
    # number of samples = num_frames
    
    N = 1024
    hop_size = 256
    
    d = 3     # degree for cubic splines
    k = n - d # number of subintervals
    N = n + d # last index of knot sequence t_i, i=0,...,N
    print("k, d, n, N:  ", k, d, n, N)
    
    # arrays for building splines
    knotVals = torch.zeros(N+1)  # knot sequence
    inputVals = torch.zeros(n)   # inputs in [0,1] for interpolation
    outputVals = torch.zeros(n)  # outputs for interpolation
    
    sr = sample_rate
    # we will not use seconds, just float samples
    # start_time = start_sample / sr  # start sample in seconds
    # end_time = end_sample / sr      # end sample in seconds
    # length_in_sec = count / sr
    nsamp = num_frames
    # signal_len = nsamp / sr
    
    print("sample rate:  ", sr)
    print("length of waveform in samples:  ", nsamp)
    # print("length of waveform in seconds:  ", signal_len)
    print("length of selection in samples:  ", count)
    # print("length of selection in seconds:  ", length_in_sec)
    
    # we don't need data type conversions (like short int to float as in wavspline.py)
    # since we have waveform already as floats so just use squeeze:
    data = torch.squeeze(waveform).numpy()
    print("data.shape:  ", data.shape)
    # i^th sample value is now data[i]

    # short_times are the inputs to the signal graph, which for the purpose of regularity should be
    # all samples used for computing the piecewise linear audio graph that covers the interval [a,b].
    # One could also think of the sample sequence start_sample,...,end_sample as the "closure" of
    # [a,b] with respect to sequences of samples, ie. it is the shortest sequence of samples such
    # [start_sample, end_sample] contains [a,b]. 
    
    steps = count + 1 # size of short_times and short_data
    short_times = np.linspace(start_sample, end_sample, num=steps)
    short_data = np.zeros(steps)
    for i in range(0, count + 1) :
        short_data[i] = data[start_sample + i]
     
    # Now short_data contains the samples to be graphed as piecewise linear graph. 
    # For the spline:
    # We want to match audio data by computing input/output values using piecewise linear function
    # through the audio samples on the time interval I=[a,b].  We do this to match
    # the type of interpolation points we are choosing for the spline on the interval [0,1], by
    # first setting up k+1=n-2 subinterval endpoints uniformly on the interval I, then inserting two
    # more input values at the midpoints of outer subintervals, to get sequence interp_times.
    # Note: Generally, a, b, and all the inputs s_i can be subsample (between sample) values. 
     
    # temp_times are for the spline graph on [a,b] 
    temp_times = np.linspace(a, b, num=n-2)
    temp_incr = (b - a) / k
    temp_times[0] = a + temp_incr / 2
    temp_times[n-3] = b - temp_incr / 2
    # we will use this as the middle of the sequence of inputs s_i
    interp_times = np.zeros(n)
    for i in range(1, n-1) :
        interp_times[i] = temp_times[i-1]
    interp_times[0] = a
    interp_times[n-1] = b
    # now interp_times play the role of inputs s_i on interval I.
    # (we could also get interp_times just by scaling and shifting inputVals from[0,1])
    print("interp times:")
    print(interp_times)
     
    interp_data = np.zeros(n)
    # First need signal values at a and b as piecewise linear.
    t = a - start_sample
    interp_data[0] = (1-t) * short_data[0] + t * short_data[1] 
    t = b - short_times[count-1]
    interp_data[n - 1] = (1-t) * short_data[count-1] + t * short_data[count] 
    # Other points are computed on data graph as piecewise linear.
    # All of the times t = interp_times[i] are between some short_times[j] and short_times[j+1].
    for i in range(1, n - 1) :  # i=1 to n-2
        j = 0
        time0 = 0
        data0 = 0
        time1 = 0
        data1 = 0
        while short_times[j] < interp_times[i] :
            data0 = short_data[j]
            if j > steps - 1 :
                data1 = b
            if j < steps :
                data1 = short_data[j+1]
            time0 = short_times[j]
            if j > steps - 1 :
                time1 = b
            if j < steps :
                time1 = short_times[j+1]
            j += 1
        t = interp_times[i]
        denom = time1 - time0
        c0 = (t - time0) / denom
        c1 = (t - time1) / denom
        # (t,y) is on line between (time0,data0) and (time1,data1)
        y = c0 * data1 - c1 * data0
        interp_data[i] = y
     
    print("interp data:") 
    print(interp_data)

    # continuing with spline setup, targets for spline from audio data:
    targets = interp_data
     
    # print("targets:  ", targets)
     
    # subinterval size:
    incr = 1 / k
    # print("incr:  ", incr)
     
    # Use inputs along the interval [0,1], first using the subinterval endpoints
    # 0,1/k,2/k,...,(k-1)/k,1 (k+1 of these) and then two more values: 1/2k and 1-1/2k.
    # The idea with using these last two values is that they give a little more information near
    # the endpoints which can affect the slope at the ends when interpolating continuous data.
    inputVals[0] = 0.0
    inputVals[1] = 0.5 * incr
    inputVals[2] = incr  # after this one, just add incr to fill in between
    inputVals[n-2] = 1 - 0.5 * incr
    inputVals[n-1] = 1.0
     
    for i in range(3,n-2) :
        inputVals[i] = inputVals[i-1] + incr
    print("inputs:")
    print(inputVals)
     
    for i in range(N+1) :
        knotVals[i] = (i-d) * incr
        if (i < d) :
            knotVals[i] = 0
        if (i > N-d-1) :
            knotVals[i] = 1
    print("knots:") 
    print(knotVals)
     
    # In previous version we assumed bcoeffs c[0]=0=c[n-1], but now we allow these to be nonzero.
    # Since they still give the function value at the ends, we can simply set c[0]=data[0] and 
    # c[n-1]=data[n-1], then need to solve for remaining c[i].  Also, the points will not all
    # be uniformly spaced, since we use k+1 uniform on [0,1], then two more at 1/2k and 1-1/2k.
    
    # Next, set up the linear system coefficient matrix A to solve for B-spline coeffs
    # using B-spline evaluation function.
    
    # Linear system rows i, columns j to solve for c[0] ... c[n-1] so
    # the entry A[i,j] should be B^3_j(s_i) for input s_i. 
    
    # the system, to solve for c_0,...,c_{n-1} 
    # for function f(t) = sum of c_j*B^3_j(t) j=0,...,n-1
    # looks like:
    # row 0: [B^3_0(s_0) B^3_1(s_0) ... B^3_{n-1}(s_0)]
    # row 1: [B^3_0(s_1) B^3_1(s_1) ... B^3_{n-1}(s_1)]
    # where s_0=0, s_1=1/2k, s_2=1/k, s_3=2/k, ... , s_{n-3}=1-1/k, s_{n-2}=1-1/2k and s_{n-1}=1
    # row n-1: [B^3_0(s_{n-1}) B^3_1(s_{n-1}) ... B^3_{n-1}(s_{n-1})]
     
    A = torch.zeros(n, n)
    # print("matrix A =  ", A)
    
    for i in range(n) :
        for j in range(n) :
            A[i, j] = newBsplineVal(3, k, j, inputVals[i])
    # print("matrix A =  ", A)
     
    # print("type(targets):  ", type(targets))
     
    B = torch.from_numpy(targets).float()
    # print("torch.dtype(b):  ", torch.dtype(b))
     
    # Now need to solve A*c=b for c (bcoeffs vector c) with b = targets
    c = torch.linalg.solve(A, B)
    print("bcoeffs vector c")
    print(c)
     
    # Next graph spline function (xvals, yvals) with the computed coefficients using 1000 points.
     
    xvals = np.linspace(start=0.0, stop=1.0, num=1000)
    # print(xvals)
    # print("size of xvals:  ", xvals.size)
    yvals = np.zeros(1000)
    for i in range(1000) :
        t = xvals[i]
        yvals[i] = computeSplineVal(d, k, c, t)
     
    # print(yvals)
    # print("size of yvals:  ", yvals.size)
     
    # rescale xvals for spline plot
    xvals = np.linspace(a, b, num=1000) # 1000 points for smooth-looking spline plot
     
    plt.figure(figsize=(15,8))
    # plt.plot(times, data)
    plt.plot(short_times, short_data, '0.8') # this defaults to piecewise linear, 0.8 is light grey
    plt.plot(xvals, yvals, 'g')  # spline plot, g = green
    title = "segment: samples " + str(start_sample)
    title += " to " + str(end_sample)
    title += " piecewise linear in grey, spline in green"
    title += " with n = " + str(n)
    plt.title(title)
    plt.ylabel("sample values")
    plt.xlabel("time in samples")
    # plt.xlim(0, signal_len)
    plt.xlim(start_sample, end_sample)
    plt.plot(interp_times, interp_data, 'ro')
    plt.show()
    
    
path = "../audio/"
file = "left.wav"
path += file
waveform, sample_rate = torchaudio.load(path)
np_waveform = waveform.numpy()
num_channels, num_frames = np_waveform.shape
print("audio file loaded: ", file)
print("sample rate: ", sample_rate)
print("length in samples: ", num_frames)
num_segments = 16
print("splitting file into ", num_segments, " segments")
segments = torch.tensor_split(waveform, num_segments, dim=1)
for j in range(num_segments) :
    print("segment: ", j, " length in samples: ", segments[j].shape)
current_segment = 3
print("processing segment: ", current_segment)
waveform = segments[current_segment]

a = 200.5
b = 350.2
n = 30
plotSegmentSpline(waveform, a, b, n)



