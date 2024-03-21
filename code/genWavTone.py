
# ----- Brief Description -----
# 
# Generate waveform given fundamental frequency f0 and key cycles using cycle interpolation. 
# genWavTone() returns waveform as tensor, insertWavTone writes into larger waveform tensor.
# inputs:  f0 = fundamental frequency, sample_rate = sample rate
# time = waveform duration in seconds,
# key_bcoeffs = B-spline coefficients vectors of each key cycle,
# keys = indices of key cycles
# gains = scalar multipliers for each key cycle, for envelope
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
# assume a and b are time values in samples between integer points, so that the
# spline is computed on interval [a,b] and evaluated at M = floor(b)-floor(a) integer 
# points or samples to produce waveform sample values. The waveform will have
# values indexed 0 to M-1 at given sample_rate. 
#
# ----- ----- ----- ----- -----

import torch
import torchaudio
import numpy as np
import math

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 
from genCycle import genCycle, insertCycle, insertCycle2, insertCycle3


def genWavTone2(f0, time, sample_rate, key_bcoeffs, knotVals, keys, gains, interp_method) :

    # inputs:
    # f0 - fundamental frequency in Hz
    # time - in decimal seconds 
    # sample_rate - integer typically 16000 or 48000 or 44100
    # key_bcoeffs - tensor (m by n matrix) of (row) vectors of bcoeffs for each cycle
    # keys - integers which give the placement of key cycles within the sequence of all cycles
    # (keys need to be less than the total number of cycles for the predicted time) 
    # gains - a sequence of scalars to multiply each key cycle by 
    # (the previous three arrays all have the same size)
    # interp_method - for cycle interpolation, 0 = none (constant), 1 = linear, 2 = quadratic, 3 = cubic
    
    # output:
    # tensor of floats as output sample values

    # the default method for cycle interpolation should be linear interpolation (of bcoeffs)
    # 0 = none (constant) should mean we are just repeating the same cycle until next key cycle.

    n = key_bcoeffs.size(dim=1) # dimension of splines for each cycle
    num_keys = len(keys)
    last_key = keys[-1]
    num_cycles = int(f0 * time) # number of cycles in final waveform
    # frac_cycle = f0 * time - num_cycles # length of last partial cycle
    if (len(key_bcoeffs) != num_keys) :
        print("inconsistent number of keys and key_bcoeffs")
    if (len(gains) != num_keys) :
        print("inconsistent number of keys and gains")
    if (last_key > int(f0 * time)) :
        print("last key cycle exceeds number of cycles predicted by f0")

    # Next multiply key cycles' bcoeffs by gains
#    for i in range(num_keys) :
#        for j in range(n) :
#            key_bcoeffs[i][j] *= gains[i]

    # Need to interpolate to fill in bcoeffs for intermediate cycles
    # Start with bcoeffs = first vector of key_bcoeffs
    all_bcoeffs = torch.zeros(num_cycles,n)  # rows are bcoeffs of each cycle
    all_bcoeffs[0] = key_bcoeffs[0]  # first row
    for i in range(num_keys) :
        key = keys[i]
        all_bcoeffs[key] = key_bcoeffs[i]  # assign entire row 
    # print(all_bcoeffs)
    # now the key bcoeffs are filled into the array all_bcoeffs 

    # keys are an increasing sequence of indices of key cycles which satisfy: 
    # keys[0]=0, keys[num_keys-1]=last_key, and 0 < keys[i] < last_key  
    # example: keys = [0,5,10,20,50,100,200,300], f0 = 440, time = 1, could be a one second
    # audio sample with 8 key cycles to be interpolated at 301-8=293 intermediate cycles
    # and 440-301=139 trailing cycles that ramp down to zero. 
    
    num_cycles = int(f0 * time) # number of whole cycles in final waveform
    print("num_cycles: ", num_cycles)
    cycle_length = sample_rate / f0 # cycle length in samples
    print("cycle length in samples:  ", cycle_length)
    waveform_length = int(sample_rate * time)
    print("waveform length: ", waveform_length)

    waveform = torch.zeros(waveform_length)  # final output samples as 1-dim tensor array

    # Now fill in all_bcoeffs using interpolation between key cycles using tensor operations 
    # on rows of all_bcoeffs, where each row is the set of n bcoeffs for one cycle, 
    # so we are basically doing row ops on a matrix
    for i in range(num_keys-1) :  # i=0...num_keys-2
        # interpolate between key_bcoeffs[i] and key_bcoeffs[i+1]
        # number of intermediate cycles (rows of bcoeffs) is keys[i+1]-keys[i]-1
        num_interm = keys[i+1]-keys[i]-1
        for j in range(num_interm) :
            start_interm = keys[i]+1  # each intermediate index is now start_interm + j
            # p goes from 1/num_interm to 1=num_interm/num_interm, so p = (j+1)/num_interm 
            p = float(j+1) / num_interm
            all_bcoeffs[start_interm + j] = (1-p) * key_bcoeffs[i] + p * key_bcoeffs[i+1]

    # Now all_bcoeffs should have all intermediate cycles filled 
    print("all_bcoeffs: ", all_bcoeffs)

    # Next, loop through all cycles and write output samples for each cycle
    # into waveform data array. 

    # number of trailing cycles (after last key cycle) is num_cycles - last_key
    num_trailing_cycles = num_cycles - last_key

    # Need to write one cycle at a time, using endpoints a and b (between samples) 
    # except for cycle 0 which is key_cycle 0 and which starts at sample 0

    a = 0.0
    b = 0.0
    count = 0 # index for output sample data
    tail = 1
    # write cycles 
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        if i < last_key :
            bcoeffs = all_bcoeffs[i]
        else :
            numer = float(num_cycles - i)
            tail = float(numer / num_trailing_cycles)
            bcoeffs = tail * all_bcoeffs[last_key]

        # previously used genCycle ...
        # cycle_data = genCycle(cycle, bcoeffs)  # this is now a torch tensor with length = number of samples
        # m = cycle_data.size(dim=0)

        insertCycle2(waveform, cycle, bcoeffs, knotVals)

        # for j in range(m) :
        #    data[count] = tail * cycle_data[j]
        #     count += 1
        #     if (count > waveform_length) :
        #        print("count exceeds waveform_length by: ", count - waveform_length)
           
    # fill last partial cycle with zeros (not necessary since using torch.zeros)
    # while count < waveform_length
    #     data[count] = 0.0

    return waveform


def genWavTone(f0, time, sample_rate, key_bcoeffs, keys, gains, interp_method) :

    # inputs:
    # f0 - fundamental frequency in Hz
    # time - in decimal seconds 
    # sample_rate - integer typically 16000 or 48000 or 44100
    # key_bcoeffs - tensor (m by n matrix) of (row) vectors of bcoeffs for each cycle
    # keys - integers which give the placement of key cycles within the sequence of all cycles
    # (keys need to be less than the total number of cycles for the predicted time) 
    # gains - a sequence of scalars to multiply each key cycle by 
    # (the previous three arrays all have the same size)
    # interp_method - for cycle interpolation, 0 = none, 1 = linear, 2 = quadratic, 3 = cubic
    
    # output:
    # tensor of floats as output sample values

    # the default method for cycle interpolation should be linear interpolation (of bcoeffs)
    # 0 = none should mean we are just repeating the same cycle until next key cycle.

    n = key_bcoeffs.size(dim=1) # dimension of splines for each cycle
    num_keys = len(keys)
    last_key = keys[-1]
    num_cycles = int(f0 * time) # number of cycles in final waveform
    # frac_cycle = f0 * time - num_cycles # length of last partial cycle
    if (len(key_bcoeffs) != num_keys) :
        print("inconsistent number of keys and key_bcoeffs")
    if (len(gains) != num_keys) :
        print("inconsistent number of keys and gains")
    if (last_key > int(f0 * time)) :
        print("last key cycle exceeds number of cycles predicted by f0")

    # Next multiply key cycles' bcoeffs by gains
#    for i in range(num_keys) :
#        for j in range(n) :
#            key_bcoeffs[i][j] *= gains[i]

    # Need to interpolate to fill in bcoeffs for intermediate cycles
    # Start with bcoeffs = first vector of key_bcoeffs
    all_bcoeffs = torch.zeros(num_cycles,n)  # rows are bcoeffs of each cycle
    all_bcoeffs[0] = key_bcoeffs[0]  # first row
    for i in range(num_keys) :
        key = keys[i]
        all_bcoeffs[key] = key_bcoeffs[i]  # assign entire row 
    print(all_bcoeffs)
    # now the key bcoeffs are filled into the array all_bcoeffs 

    # keys are an increasing sequence of indices of key cycles which satisfy: 
    # keys[0]=0, keys[num_keys-1]=last_key, and 0 < keys[i] < last_key  
    # example: keys = [0,5,10,20,50,100,200,300], f0 = 440, time = 1, could be a one second
    # audio sample with 8 key cycles to be interpolated at 301-8=293 intermediate cycles
    # and 440-301=139 trailing cycles that ramp down to zero. 
    
    num_cycles = int(f0 * time) # number of whole cycles in final waveform
    print("num_cycles: ", num_cycles)
    cycle_length = sample_rate / f0 # cycle length in samples
    print("cycle length in samples:  ", cycle_length)
    waveform_length = int(sample_rate * time)
    print("waveform length: ", waveform_length)

    waveform = torch.zeros(waveform_length)  # final output samples as 1-dim tensor array

    # Now fill in all_bcoeffs using interpolation between key cycles using tensor operations 
    # on rows of all_bcoeffs, where each row is the set of n bcoeffs for one cycle, 
    # so we are basically doing row ops on a matrix
    for i in range(num_keys-1) :  # i=0...num_keys-2
        # interpolate between key_bcoeffs[i] and key_bcoeffs[i+1]
        # number of intermediate cycles (rows of bcoeffs) is keys[i+1]-keys[i]-1
        num_interm = keys[i+1]-keys[i]-1
        for j in range(num_interm) :
            start_interm = keys[i]+1  # each intermediate index is now start_interm + j
            # p goes from 1/num_interm to 1=num_interm/num_interm, so p = (j+1)/num_interm 
            p = float(j+1) / num_interm
            all_bcoeffs[start_interm + j] = (1-p) * key_bcoeffs[i] + p * key_bcoeffs[i+1]

    # Now all_bcoeffs should have all intermediate cycles filled 
    print("all_bcoeffs: ", all_bcoeffs)

    # Next, loop through all cycles and write output samples for each cycle
    # into waveform data array. 

    # number of trailing cycles (after last key cycle) is num_cycles - last_key
    num_trailing_cycles = num_cycles - last_key

    # Need to write one cycle at a time, using endpoints a and b (between samples) 
    # except for cycle 0 which is key_cycle 0 and which starts at sample 0

    a = 0.0
    b = 0.0
    count = 0 # index for output sample data
    tail = 1
    # write cycles 
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        if i < last_key :
            bcoeffs = all_bcoeffs[i]
        else :
            numer = float(num_cycles - i)
            tail = float(numer / num_trailing_cycles)
            bcoeffs = tail * all_bcoeffs[last_key]

        # previously used genCycle ...
        # cycle_data = genCycle(cycle, bcoeffs)  # this is now a torch tensor with length = number of samples
        # m = cycle_data.size(dim=0)

        insertCycle(waveform, cycle, bcoeffs)

        # for j in range(m) :
        #    data[count] = tail * cycle_data[j]
        #     count += 1
        #     if (count > waveform_length) :
        #        print("count exceeds waveform_length by: ", count - waveform_length)
           
    # fill last partial cycle with zeros (not necessary since using torch.zeros)
    # while count < waveform_length
    #     data[count] = 0.0

    return waveform

# We need a function which takes input: f0, time (duration of tone in sec), 
# and computes a number of cycles and also an appropriate sequence of key cycles.
# For example, if there are only 6 cycles, then they should all be key cycles.

def getKeys(f0, time) :
    num_cycles = int(f0 * time)
    if num_cycles < 8 :
        keys = torch.zeros(num_cycles)
        for i in range(num_cycles) :
            keys[i] = i
        return keys
    keys = torch.tensor([0,10,20,30,50,70,90])

# This function inserts a tone in the tensor waveform:

def insertWavTone(waveform, start_time, f0, time, sample_rate, key_bcoeffs, keys, gains, interp_method) :

    # this is same function as genWavTone but instead of returning the waveform of the tone
    # it inserts those samples into a bigger waveform at a particular start sample.
    # This way it functions like insertCycle as compared to genCycle. 

    # inputs:
    # f0 - fundamental frequency in Hz
    # time - length of tone in decimal seconds 
    # start_time - starting time in waveform to insert tone, in decimal samples 
    # sample_rate - integer typically 16000 or 48000 or 44100
    # key_bcoeffs - tensor (m by n matrix) of m (rows) of n bcoeffs for each cycle
    # keys - integers which give the placement of key cycles within the sequence of all cycles
    # (keys need to be less than the total number of cycles for the predicted time) 
    # gains - a sequence of scalars to multiply each key cycle by 
    # (the previous three arrays all have the same size)
    # interp_method - for cycle interpolation, 0 = none, 1 = linear, 2 = quadratic, 3 = cubic
    
    # output: (none)
    # sample values inserted into waveform in place

    # the default method for cycle interpolation should be linear interpolation (of bcoeffs)
    # 0 = none should mean we are just repeating the same cycle until next key cycle.

    n = key_bcoeffs.size(dim=1) # dimension of splines for each cycle
    num_keys = len(keys)
    last_key = int(keys[-1])
    num_cycles = int(f0 * time) # number of cycles in final waveform
    # frac_cycle = f0 * time - num_cycles # length of last partial cycle
    if (len(key_bcoeffs) != num_keys) :
        print("inconsistent number of keys and key_bcoeffs")
    if (len(gains) != num_keys) :
        print("inconsistent number of keys and gains")
    if (last_key > int(f0 * time)) :
        print("last key cycle exceeds number of cycles predicted by f0")

    # Next multiply key cycles' bcoeffs by gains
    for i in range(num_keys) :
        key_bcoeffs[i] *= gains[i]
    # key_bcoeffs *= gains

    # Need to interpolate to fill in bcoeffs for intermediate cycles
    # Start with bcoeffs = first vector of key_bcoeffs
    all_bcoeffs = torch.zeros(num_cycles,n)  # rows are bcoeffs of each cycle
    all_bcoeffs[0] = key_bcoeffs[0]  # first row
    for i in range(num_keys) :
        key = int(keys[i])
        all_bcoeffs[key] = key_bcoeffs[i]  # assign entire row 
    # print(all_bcoeffs)
    # now the key bcoeffs are filled into the array all_bcoeffs 

    num_cycles = int(f0 * time) # number of whole cycles in final waveform
    print("num_cycles: ", num_cycles)
    cycle_length = sample_rate / f0 # cycle length in samples
    print("cycle length in samples:  ", cycle_length)
    waveform_length = int(sample_rate * time)
    print("waveform length: ", waveform_length)

    # Now fill in all_bcoeffs using interpolation between key cycles using tensor operations 
    # on rows of all_bcoeffs, where each row is the set of n bcoeffs for one cycle, 
    # so we are basically doing row ops on a matrix
    for i in range(num_keys-1) :  # i=0...num_keys-2
        # interpolate between key_bcoeffs[i] and key_bcoeffs[i+1]
        # number of intermediate cycles (rows of bcoeffs) is keys[i+1]-keys[i]-1
        num_interm = int(keys[i+1]-keys[i]-1)
        for j in range(num_interm) :
            start_interm = int(keys[i])+1  # each intermediate index is now start_interm + j
            # p goes from 1/num_interm to 1=num_interm/num_interm, so p = (j+1)/num_interm 
            p = float(j+1) / num_interm
            all_bcoeffs[start_interm + j] = (1-p) * key_bcoeffs[i] + p * key_bcoeffs[i+1]

    # Now all_bcoeffs should have all intermediate cycles filled 
    # print("all_bcoeffs: ", all_bcoeffs)

    # number of trailing cycles (after last key cycle) is num_cycles - last_key
    num_trailing_cycles = num_cycles - last_key

    a = start_time
    b = start_time
    # count = 0 # index for output sample data
    tail = 1
    # write cycles into waveform from start_time to end of cycles
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        if i < last_key :
            bcoeffs = all_bcoeffs[i]
        else :
            numer = float(num_cycles - i)
            tail = float(numer / num_trailing_cycles)
            bcoeffs = tail * all_bcoeffs[last_key]
        # now insert cycle into waveform from start_time to end of cycle
        # insertCycle(waveform, cycle, bcoeffs)
        insertCycle(waveform, cycle, bcoeffs)


def insertWavTone2(waveform, start_time, f0, time, sample_rate, key_bcoeffs, knotVals, keys, gains, interp_method) :

    # this is same function as genWavTone but instead of returning the waveform of the tone
    # it inserts those samples into a bigger waveform at a particular start sample.
    # This way it functions like insertCycle as compared to genCycle. 

    # inputs:
    # f0 - fundamental frequency in Hz
    # time - length of tone in decimal seconds 
    # start_time - starting time in waveform to insert tone, in decimal samples 
    # sample_rate - integer typically 16000 or 48000 or 44100
    # key_bcoeffs - tensor (m by n matrix) of (row) vectors of bcoeffs for each cycle
    # knotVals - tensor of knots for spline evaluation
    # keys - integers which give the placement of key cycles within the sequence of all cycles
    # (keys need to be less than the total number of cycles for the predicted time) 
    # gains - a sequence of scalars to multiply each key cycle by 
    # (the previous three arrays all have the same size)
    # interp_method - for cycle interpolation, 0 = none, 1 = linear, 2 = quadratic, 3 = cubic
    
    # output: (none)
    # sample values inserted into waveform in place

    # the default method for cycle interpolation should be linear interpolation (of bcoeffs)
    # 0 = none should mean we are just repeating the same cycle until next key cycle.

    n = key_bcoeffs.size(dim=1) # dimension of splines for each cycle
    num_keys = len(keys)
    last_key = int(keys[-1])
    num_cycles = int(f0 * time) # number of cycles in final waveform
    # frac_cycle = f0 * time - num_cycles # length of last partial cycle
    if (len(key_bcoeffs) != num_keys) :
        print("inconsistent number of keys and key_bcoeffs")
    if (len(gains) != num_keys) :
        print("inconsistent number of keys and gains")
    if (last_key > int(f0 * time)) :
        print("last key cycle exceeds number of cycles predicted by f0")

    new_bcoeffs = torch.zeros(num_keys, n)
    for m in range(num_keys) :
        temp = torch.tensor(key_bcoeffs[m])
        new_bcoeffs[m] = temp

    # Next multiply key cycles' bcoeffs by gains
    # print("GAINS:  ", gains)
    # print("KEY_BCOEFFS:  ", key_bcoeffs)
    for i in range(num_keys) :
        new_bcoeffs[i] *= gains[i]
    # key_bcoeffs *= gains

      
    # Need to interpolate to fill in bcoeffs for intermediate cycles
    # Start with bcoeffs = first vector of key_bcoeffs
    all_bcoeffs = torch.zeros(num_cycles,n)  # rows are bcoeffs of each cycle
    all_bcoeffs[0] = new_bcoeffs[0]  # first row
    old_key = -1
    for i in range(num_keys) :
        key = int(keys[i])
        if key < num_cycles :
            if key > old_key :
                all_bcoeffs[key] = new_bcoeffs[i]  # assign entire row 
                old_key = key
    # print(all_bcoeffs)
    # now the key bcoeffs are filled into the array all_bcoeffs 

    num_cycles = int(f0 * time) # number of whole cycles in final waveform
    print("num_cycles: ", num_cycles)
    cycle_length = sample_rate / f0 # cycle length in samples
    print("cycle length in samples:  ", cycle_length)
    waveform_length = int(sample_rate * time)
    print("waveform length: ", waveform_length)

    # Now fill in all_bcoeffs using interpolation between key cycles using tensor operations 
    # on rows of all_bcoeffs, where each row is the set of n bcoeffs for one cycle, 
    # so we are basically doing row ops on a matrix
    for i in range(num_keys-1) :  # i=0...num_keys-2
        # interpolate between key_bcoeffs[i] and key_bcoeffs[i+1]
        # number of intermediate cycles (rows of bcoeffs) is keys[i+1]-keys[i]-1
        num_interm = int(keys[i+1]-keys[i]-1)
        # if num_interm == 0 then do nothing
        for j in range(num_interm) :
            start_interm = int(keys[i])+1  # each intermediate index is now start_interm + j
            # p goes from 1/num_interm to 1=num_interm/num_interm, so p = (j+1)/num_interm 
            p = float(j+1) / num_interm
            # linear interpolation between key_bcoeffs rows with ratio p: 
            all_bcoeffs[start_interm + j] = (1-p) * new_bcoeffs[i] + p * new_bcoeffs[i+1]

    # Now all_bcoeffs should have all intermediate cycles filled 
    # print("all_bcoeffs: ", all_bcoeffs)

    # number of trailing cycles (after last key cycle) is num_cycles - last_key
    num_trailing_cycles = num_cycles - last_key

    a = start_time
    b = start_time
    # count = 0 # index for output sample data
    tail = 1
    # write cycles into waveform from start_time to end of cycles
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        if i < last_key :
            bcoeffs = all_bcoeffs[i]
        else :
            numer = float(num_cycles - i)
            # removing next tail calculation to see how it affects envelope
            # tail = float(numer / num_trailing_cycles)
            bcoeffs = tail * all_bcoeffs[last_key]
        # now insert cycle into waveform from start_time to end of cycle
        # insertCycle(waveform, cycle, bcoeffs)
        insertCycle2(waveform, cycle, bcoeffs, knotVals)


def insertWavTone3(waveform, voice, start_time, f0, time, sample_rate, key_bcoeffs, knotVals, keys, gains, interp_method) :

    # this is same function as insertWavTone2 but now also uses voice.

    # inputs:
    # waveform is now array of waveforms, one for each voice
    # voice is integer voice number or index 0,...,voices-1 
    # f0 - fundamental frequency in Hz
    # time - length of tone in decimal seconds 
    # start_time - starting time in waveform to insert tone, in decimal samples 
    # sample_rate - integer typically 16000 or 48000 or 44100
    # key_bcoeffs - tensor (m by n matrix) of (row) vectors of bcoeffs for each cycle
    # knotVals - tensor of knots for spline evaluation
    # keys - integers which give the placement of key cycles within the sequence of all cycles
    # (keys need to be less than the total number of cycles for the predicted time) 
    # gains - a sequence of scalars to multiply each key cycle by 
    # (the previous three arrays all have the same size)
    # interp_method - for cycle interpolation, 0 = none, 1 = linear, 2 = quadratic, 3 = cubic
    
    # output: (none)
    # sample values inserted into waveform in place

    # the default method for cycle interpolation should be linear interpolation (of bcoeffs)
    # 0 = none should mean we are just repeating the same cycle until next key cycle.

    n = key_bcoeffs.size(dim=1) # dimension of splines for each cycle
    num_keys = len(keys)
    last_key = int(keys[-1])
    num_cycles = int(f0 * time) # number of cycles in final waveform
    # frac_cycle = f0 * time - num_cycles # length of last partial cycle
    if (len(key_bcoeffs) != num_keys) :
        print("inconsistent number of keys and key_bcoeffs")
    if (len(gains) != num_keys) :
        print("inconsistent number of keys and gains")
    if (last_key > int(f0 * time)) :
        print("last key cycle exceeds number of cycles predicted by f0")

    new_bcoeffs = torch.zeros(num_keys, n)
    for m in range(num_keys) :
        temp = torch.tensor(key_bcoeffs[m])
        new_bcoeffs[m] = temp

    # Next multiply key cycles' bcoeffs by gains
    # print("GAINS:  ", gains)
    # print("KEY_BCOEFFS:  ", key_bcoeffs)
    for i in range(num_keys) :
        new_bcoeffs[i] *= gains[i]
    # key_bcoeffs *= gains

      
    # Need to interpolate to fill in bcoeffs for intermediate cycles
    # Start with bcoeffs = first vector of key_bcoeffs
    all_bcoeffs = torch.zeros(num_cycles,n)  # rows are bcoeffs of each cycle
    all_bcoeffs[0] = new_bcoeffs[0]  # first row
    old_key = -1
    for i in range(num_keys) :
        key = int(keys[i])
        if key < num_cycles :
            if key > old_key :
                all_bcoeffs[key] = new_bcoeffs[i]  # assign entire row 
                old_key = key
    # print(all_bcoeffs)
    # now the key bcoeffs are filled into the array all_bcoeffs 

    num_cycles = int(f0 * time) # number of whole cycles in final waveform
    print("num_cycles: ", num_cycles)
    cycle_length = sample_rate / f0 # cycle length in samples
    print("cycle length in samples:  ", cycle_length)
    waveform_length = int(sample_rate * time)
    print("waveform length: ", waveform_length)

    # Now fill in all_bcoeffs using interpolation between key cycles using tensor operations 
    # on rows of all_bcoeffs, where each row is the set of n bcoeffs for one cycle, 
    # so we are basically doing row ops on a matrix
    for i in range(num_keys-1) :  # i=0...num_keys-2
        # interpolate between key_bcoeffs[i] and key_bcoeffs[i+1]
        # number of intermediate cycles (rows of bcoeffs) is keys[i+1]-keys[i]-1
        num_interm = int(keys[i+1]-keys[i]-1)
        # if num_interm == 0 then do nothing
        for j in range(num_interm) :
            start_interm = int(keys[i])+1  # each intermediate index is now start_interm + j
            # p goes from 1/num_interm to 1=num_interm/num_interm, so p = (j+1)/num_interm 
            p = float(j+1) / num_interm
            # linear interpolation between key_bcoeffs rows with ratio p: 
            all_bcoeffs[start_interm + j] = (1-p) * new_bcoeffs[i] + p * new_bcoeffs[i+1]

    # Now all_bcoeffs should have all intermediate cycles filled 
    # print("all_bcoeffs: ", all_bcoeffs)

    # number of trailing cycles (after last key cycle) is num_cycles - last_key
    num_trailing_cycles = num_cycles - last_key

    a = start_time
    b = start_time
    # count = 0 # index for output sample data
    tail = 1
    # write cycles into waveform from start_time to end of cycles
    for i in range(num_cycles) :
        # write cycle i
        a = b
        b = a + cycle_length
        b = reset(b)
        cycle = [a, b]
        if i < last_key :
            bcoeffs = all_bcoeffs[i]
        else :
            numer = float(num_cycles - i)
            # removing next tail calculation to see how it affects envelope
            # tail = float(numer / num_trailing_cycles)
            bcoeffs = tail * all_bcoeffs[last_key]
        # now insert cycle into waveform from start_time to end of cycle
        # insertCycle(waveform, cycle, bcoeffs)
        insertCycle3(waveform, voice, cycle, bcoeffs, knotVals)


def reset(b) :  # forces a and b to avoid exact integer sample values
    newb = b
    if abs(b - math.floor(b)) < 0.01 :
        newb = math.floor(b) - 0.01
    if abs(b - math.ceil(b)) < 0.01 :
        newb = math.ceil(b) - 0.01
    return newb

# unused functions:

# def key_floor(i) :
#     if iskey(i) :
#         return i
#     j = i - 1
#     while True :
#         if iskey(j) :
#             return j
#         j = j - 1
#         if j < 0 :
#             return -1
# 
# def iskey(i) :
#     for j in keys :
#         if (j == i) :
#             return True
#     return False
# 
# def index_of_key(i) :
#     for j in range(num_keys) :
#         if (i == keys[j]) :
#             return j
#     return -1
# 
