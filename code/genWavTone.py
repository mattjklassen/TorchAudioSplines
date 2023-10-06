
# ----- Brief Description -----
# 
# Generate waveform given fundamental frequency f0 and key cycles
# using cycle interpolation, and return waveform.
# inputs: 
# f0 = fundamental frequency
# sample_rate = sample rate
# key_bcoeffs = B-spline coefficients vectors of each key cycle 
# keys = indices of key cycle
# gains = scalar multipliers for each key cycle, for envelope
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
from genCycle import genCycle


def genWavTone(f0, time, sample_rate, key_bcoeffs, keys, gains, interp_method) :

    # f0 - fundamental frequency in Hz
    # time - in decimal seconds 
    # sample_rate - integer typically 16000 or 48000 or 44100
    # key_bcoeffs - tensor or list of vectors of bcoeffs for each cycle
    # keys - integers which give the placement of key cycles within the sequence of all cycles
    # (keys need to fit into the number of cycles for the predicted time) 
    # gains - a sequence of scalars to multiply each key cycle by 
    # (the previous three arrays all have the same size)
    # interp_method - for cycle interpolation, 0 = none, 1 = linear, 2 = quadratic, 3 = cubic
    
    # the default method should be linear interpolation
    # 0 = none should mean just repeating the same cycle until next key cycle.

    n = len(key_bcoeffs[0])  # dimension of splines for one cycle
    data = []
    num_keys = len(keys)
    M = keys[-1]
    num_cycles = int(f0 * time) # number of cycles in final waveform
    frac_cycle = f0 * time - num_cycles # length of last partial cycle
    if (len(key_cycles) != num_keys) :
        print("inconsistent number of keys and key_cycles")
        return data[]
    if (len(key_bcoeffs) != num_keys) :
        print("inconsistent number of keys and key_bcoeffs")
        return data[]
    if (len(gains) != num_keys) :
        print("inconsistent number of keys and gains")
        return data[]
    if (M > int(f0 * time)) :
        print("last key cycle exceeds number of cycles predicted by f0")
        return data[]

    # Next interpolate to fill in bcoeffs for intermediate cycles
    # Start with bcoeffs = first vector of key_bcoeffs
    bcoeffs = key_bcoeffs[0]

    # keys are an increasing sequence of indices of key cycles which satisfy: 
    # keys[0]=0, keys[num_keys-1]=M, and 0 < keys[i] < M  
    # example: keys = [0,5,10,20,50,100,200,300,400,500] could be a one second
    # audio sample with 10 key cycles to be interpolated at 501-10=491 intermediate cycles
    # producing a waveform with f0 = 501 Hz and cycles of length sample_rate / 501 samples.
    
    num_cycles = int(f0 * time) # number of whole cycles in final waveform
    cycle_length = sample_rate / num_cycles

    for i in range(num_keys-1) :  # i=0...num_keys-2
        # interpolate between key_bcoeffs[i] and key_bcoeffs[i+1]
        # number of intermediate cycles is keys[i+1]-keys[i]-1
        # number of trailing cycles (after last key cycle) is num_cycles - M
        num_interm = keys[i+1]-keys[i]-1
        for j in range(num_interm) :
            start_interm = keys[i]+1  # each intermediate index is now start_interm + j
            bcoeffs = []  # temp list for intermediate bcoeffs array
            bcoeffs0 = key_bcoeffs[i]   
            bcoeffs1 = key_bcoeffs[i+1]
            for k in range(n) :
                b0 = bcoeffs0[k]
                b1 = bcoeffs1[k]
                increment = (b1 - b0) / (num_interm + 1)
                bcoeffs[k] = b0 + increment * (j+1)
            all_bcoeffs.append(bcoeffs)
        all_bcoeffs.append(key_bcoeffs[i+1])

    # Now all_bcoeffs should have all cycles filled 
    # Next, loop through all cycles and write output samples for each cycle
    # into one continuous waveform data array. 

    def key_floor(i) :
        if iskey(i) :
            return i
        j = i - 1
        while True :
            if iskey(j) :
                return j
            j = j - 1
            if j < 0 :
                return -1

    def iskey(i) :
        for j in keys :
            if (j == i) :
                return True
        return False

    def index_of_key(i) :
        for j in range(num_keys) :
            if (i == keys[j]) :
                return j
        return -1

    def reset(b) :
        newb = b
        if abs(b - math.floor(b)) < 0.01 :
            newb = math.floor(b) - 0.01
        if abs(b - math.ceil(b)) < 0.01 :
            newb = math.ceil(b) - 0.01
        return newb

    J = 0 # index of big loop on sample data
    # Need to write one cycle at a time, using endpoints a and b (between samples) 
    # except for cycle 0 which is key_cycle 0 and which starts at sample 0

    # write cycle 0:
    a = 0, b = cycle_length
    cycle = [a, b]
    bcoeffs = key_bcoeffs[0]
    cycle_data = genCycle(cycle, bcoeffs)
    m = len(cycle_data)
    for j in range(m) :
        data.append(cycle_data[j]
    b = reset(b)

    # write other cycles for i > 0 and then last partial cycle:
    i = 1
    while (i < num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        if iskey(i) :
            bcoeffs = key_bcoeffs[index_of_key(i)]
        else :
            j = key_foor(i)  # index of key cycle less than i
            # now current cycle i is between key cycles keys[j] and keys[j+1]
        # interpolate between key_bcoeffs[i] and key_bcoeffs[i+1]
        # number of intermediate cycles is keys[i+1]-keys[i]-1
        # number of trailing cycles (after last key cycle) is num_cycles - M
        num_interm = keys[i+1]-keys[i]-1
        for j in range(num_interm) :
            start_interm = keys[i]+1  # each intermediate index is now start_interm + j
            bcoeffs = []  # temp list for intermediate bcoeffs array
            bcoeffs0 = key_bcoeffs[i]   
            bcoeffs1 = key_bcoeffs[i+1]
            for k in range(n) :
                b0 = bcoeffs0[k]
                b1 = bcoeffs1[k]
                increment = (b1 - b0) / (num_interm + 1)
                bcoeffs[k] = b0 + increment * (j+1)
            all_bcoeffs.append(bcoeffs)
        all_bcoeffs.append(key_bcoeffs[i+1])

    # Now all_bcoeffs should have all cycles filled 



        key_data = genCycle(cycle, bcoeffs)
        m = len(key_data)
        for j in range(m) :  # unused index j
            data.append(key_data[j])
        i += 1

    else :
        # interpolate bcoeffs and write intermediate cycle 
        # for linear interpolation of bcoeffs between key cycles
        # need number of steps between keys, so use function steps(i)
        # to compute that number of steps: if keys[p] < i < keys[p+1]
        # then steps(i) = keys[p+1]-keys[p].  

    

