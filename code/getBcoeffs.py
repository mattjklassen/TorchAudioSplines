# ----- Brief Description -----
# 
# get bcoeffs (B-spline coefficients) from a cycle (or segment) [a,b] in a waveform
# inputs: waveform (tensor of audio data), cycle = [a,b], n = dimension of splines
# return: bcoeffs vector 
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# Note: we are not assuming the endpoint values are zero, so also the bcoeffs c[0]
# and c[n-1] are also not necessarily zero.
#
# ----- ----- ----- ----- -----


import torch
import torchaudio
import numpy as np
import math
from computeBsplineVal import newBsplineVal 


# Compute bcoeffs for one cycle = [a, b] using waveform data and dim=n
# where a,b are float samples, ie. time values measured in samples (typically sub-sample)
# and time zero is start of waveform, max time is last sample t_max, 0 \leq a < b \leq t_max.
# Return a list of bcoeffs of size n as PyTorch tensor.
# Note: the input waveform is processed in a few steps:

# the initial wavefrom tensor typically comes from a call like this:
# waveform, sample_rate = torchaudio.load(path)
# np_waveform = waveform.numpy() 
# num_channels, num_frames = np_waveform.shape
## i^th sample value is now np_waveform[0,i] (but becomes data below)
## number of samples = num_frames
# data = torch.squeeze(waveform).numpy()
## i^th sample value is now data[i]
# waveform can also be a segment coming from segments[i] where:
# segments = torch.split(waveform, segment_size, dim=1)


def getBcoeffs(waveform, sample_rate, cycle, n) :

    a = cycle[0]
    b = cycle[1]

    # use these two sample values to compute interpolated values at a and b: 
    start_sample = math.floor(a) 
    end_sample = math.ceil(b)
    # print("a = ", a, "  b = ", b)
    # print("start_sample = ", start_sample, "  end_sample = ", end_sample)

    # count is the number of samples used to compute all target values in [a,b]
    # and thus also to compute all the B-spline coefficients for the cycle spline.
    # The number of samples computed with the cycle spline for "resampling" is count - 2
    # since we assume that a and b are between samples.  
    count = int(end_sample - start_sample) + 1
    
    # assume waveform comes from read of this type: 
    # waveform, sample_rate = torchaudio.load("output.wav")
    # np_waveform = waveform.numpy()
    # num_channels, num_frames = np_waveform.shape
    # segments = torch.tensor_split(waveform, 16, dim=1)
    # waveform = segments[0]

    np_waveform = waveform.numpy() 
    num_channels, num_frames = np_waveform.shape
    # i^th sample value is now np_waveform[0,i] (but becomes data below)
    # number of samples = num_frames
    
    # default FFT size is 1024, hop size is 256 but may change 
    N = 1024
    hop_size = 256
    
    d = 3     # degree for cubic splines
    k = n - d # number of subintervals
    N = n + d # last index of knot sequence t_i, i=0,...,N
    # print("k, d, n, N:  ", k, d, n, N)
    
    # arrays for building splines
    knotVals = torch.zeros(N+1)
    inputVals = torch.zeros(n)
    outputVals = torch.zeros(n)
    
    sr = sample_rate
    start_time = start_sample / sr  # start sample in seconds
    end_time = end_sample / sr      # end sample in seconds
    length_in_sec = count / sr
    nsamp = num_frames
    signal_len = nsamp / sr
    
    # print("sample rate:  ", sr)
    # print("length of audio segment in samples: (nsamp) ", nsamp)
    # print("length of audio segment in seconds:  ", signal_len)
    # print("length of short selection in samples: (count) ", count)
    # print("length of short selection in seconds:  ", length_in_sec)
    
    # we don't need data type conversions (like short int to float as in wavspline.py)
    # since we have waveform already as floats so just use squeeze:
    data = torch.squeeze(waveform).numpy()
    # i^th sample value is now data[i]

    # short_times are the inputs to the signal graph, which for the purpose of regularity should be
    # all samples used for computing the piecewise linear audio graph that covers the interval [a,b].
    # One could also think of the sample sequence start_sample,...,end_sample as the "closure" of
    # [a,b] with respect to sequences of samples, ie. it is the shortest sequence of samples such
    # [start_sample, end_sample] contains [a,b]. 
    
    steps = count  # size of short_times and short_data
    # set up graph inputs in terms of samples 
    short_times = np.linspace(start_sample, end_sample, num=steps)
    short_data = np.zeros(steps)
    for i in range(0, steps) :
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
    # print("interp times:  ", interp_times)
    
    interp_data = np.zeros(n)
    # end points are NOT same as sample data
    # interp_data[0] = linear interpolation at t=a
    time0 = start_sample
    time1 = time0 + 1
    data0 = short_data[0]
    data1 = short_data[1]
    t = a
    c0 = (t - time0) # / denom
    c1 = (t - time1) # / denom
    # (t,y) is on line between (time0,data0) and (time1,data1)
    y = c0 * data1 - c1 * data0
    interp_data[0] = y
    # print("interp_data[0] = ", y)
    # interp_data[n - 1] = linear interpolation at t=b
    time0 = end_sample - 1
    time1 = end_sample
    data0 = short_data[count-2]
    data1 = short_data[count-1]
    t = b
    c0 = (t - time0) # / denom
    c1 = (t - time1) # / denom
    # (t,y) is on line between (time0,data0) and (time1,data1)
    y = c0 * data1 - c1 * data0
    interp_data[n-1] = y
    # print("interp_data[n-1] = ", y)

    # Other points are computed on data graph as piecewise linear.
    # All of the times t = interp_times[i] are between some short_times[j] and short_times[j+1].
    for i in range(1, n - 1) :  # i=1 to n-2
        j = 0
        time0 = 0
        data0 = 0
        time1 = 0
        data1 = 0
        while short_times[j] < interp_times[i] :
            # print("i, j, short_times[j], interp_times[i]:  ", i, j, sr * short_times[j], interp_times[i])
            data0 = short_data[j]
            data1 = short_data[j+1] # this should always work, not out of range
            time0 = short_times[j]
            time1 = short_times[j+1] # this should always work, not out of range
            j += 1
        t = interp_times[i]
        # denom = time1 - time0 = 1
        c0 = (t - time0) # / denom
        c1 = (t - time1) # / denom
        # (t,y) is on line between (time0,data0) and (time1,data1)
        y = c0 * data1 - c1 * data0
        interp_data[i] = y
    
    # print("interp data:  ", interp_data)
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
    # print("inputs:  ", inputVals)
    
    for i in range(N+1) :
        knotVals[i] = (i-d) * incr
        if (i < d) :
            knotVals[i] = 0
        if (i > N-d-1) :
            knotVals[i] = 1
    # print("knots:  ", knotVals)
    
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
    # print("torch.dtype(B):  ", torch.dtype(B))
    
    # Now need to solve A*c=B for c (bcoeffs vector c) with B = targets
    bcoeffs = torch.linalg.solve(A, B)
    # print("bcoeffs vector c =  ", c)
    
    return bcoeffs


# this function imports bcoeffs from text file of strings (printed floats)
# and returns a pytorch tensor of those floats.
def import_bcoeffs(file) :
    bcoeffs_str = []
    with open(file, 'r') as f:
        bcoeffs_str = f.readlines()
        f.close()
    n = len(bcoeffs_str)
    bcoeffs = torch.zeros(n)
    for i in range(n) :
        bcoeffs[i] = float(bcoeffs_str[i])
    return bcoeffs


def export_bcoeffs(file, bcoeffs) :
    bcoeffs_str = []
    for i in range(len(bcoeffs)) :
        bcoeffs_str.append(str(bcoeffs[i]))
        bcoeffs_str.append('\n')
    with open(file, 'w') as f:
        f.writelines(bcoeffs_str)
        f.close()

