
# ----- Brief Description -----
# 
# Generate one cycle as waveform, and return waveform.
# inputs: cycle = [a,b], B-spline coefficients vector (tensor) = bcoeffs.
# assume a and b are time values in samples between integer points, so that the
# spline is computed on interval [a,b] and evaluated at M = floor(b)-floor(a) integer 
# points or samples to produce waveform sample values. The waveform will have
# values indexed 0 to M-1 at given sample_rate. 
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import torch
import torchaudio
import numpy as np
import math

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 


def genCycle(cycle, bcoeffs) :

    d = 3
    n = bcoeffs.size(dim=0)
    k = n - d
    a = cycle[0]
    b = cycle[1]
    # print("generating cycle 0")
    t = 0.0
    first_sample = math.ceil(a)
    last_sample = math.floor(b)
    num_samples = last_sample - first_sample + 1
    # if b < 201 :
    #     print("num_samples: ", num_samples)
    data = torch.zeros(num_samples)
    t = first_sample
    for i in range(num_samples) :
        # evaluate splines on interval [0,1], so for any t in [a,b]
        # first transform to new t01 = (t-a)/(b-a)
        t01 = float((t - a) / (b - a))
        y = computeSplineVal(d, k, bcoeffs, t01)
        data[i] = y
        t += 1
    return data


def insertCycle(waveform, cycle, bcoeffs) :

    # This version inserts sample values for cycle into tensor array called waveform
    # at the appropriate sample values between a=cycle[0] and b=cycle[1]. 
    d = 3
    n = bcoeffs.size(dim=0)
    k = n - d
    a = cycle[0]
    b = cycle[1]
    t = 0.0
    first_sample = math.ceil(a)
    last_sample = math.floor(b)
    num_samples = last_sample - first_sample + 1
    # data = torch.zeros(num_samples)
    t = first_sample
    for i in range(num_samples) :
        # evaluate splines on interval [0,1], so for any t in [a,b]
        # first transform to new t01 = (t-a)/(b-a)
        t01 = float((t - a) / (b - a))
        y = computeSplineVal(d, k, bcoeffs, t01)
    #    data[i] = y
        waveform[first_sample + i] = y
        t += 1
    # return data

