# ----- Brief Description -----
# 
# This script tests torchyin as in the example code, by creating a tensor of the
# piano fundamental frequencies f_0 and corresponding sinusoids with those f_0
# and then running YIN f_0 prediction on those audio samples.  YIN predicts a
# period length tau by using correlation function comparisons for various values
# of tau to detect maximum values (see https://brentspell.com/2022/pytorch-yin/).
# We add to this example some output which shows that the predictions are less
# accurate as frequency increases and especially when tau is small and is at
# the midpoint between two integers. (see comments at the bottom of output)
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import torch
import torchyin
import math

torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=4)
# precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, 

sampleRate = 48000

# create tensor of piano frequencies of data type torch.float32
f = 2 ** ((torch.arange(88) - 48) / 12) * 440
fNP = f.numpy()
print("piano frequencies:")
print("f:")
print(f)

# create tensor of int64 values from 0 to sampleRate-1 to use as 1 sec of time sample values
t = torch.arange(sampleRate)
print("type of t data and type of t:")
print(t.dtype)
print(t.type())

# f.unsqueeze(1) is a "col" tensor where each value of f is now a (row) 1-dim tensor with one element
# print(f.unsqueeze(1))
# t.unsqueeze(0) is a "row" tensor where each value of t is now a (col) 1-dim tensor with one element

# Next create tensor y for input into torchyin:
# convert tensor f.unsqueeze(1) first by mult by scalar (2*Pi) then multiply by t.unsqueeze(0)/SR.
# result col * row so an outer product where each row i has freq given by f[i] multiplied by each time
# value t/SR which covers one second of time.  Each value has sin() applied to it, so each row becomes a
# signal which is sinusoidal with freq f[i] for each of the piano frequencies.  This is then basically a
# 2D array of floats y with dtype torch.float32.

y = torch.sin(2 * torch.pi * f.unsqueeze(1) / sampleRate * t.unsqueeze(0))
print("type of y data and type of y:")
print(y.dtype)
print(y.type())

pitch = torchyin.estimate(
    y,
    sample_rate=sampleRate,
    pitch_min=20,
    pitch_max=5000,
)

p = pitch[:, 0]

print("pitches computed with YIN:")
print("p:")
print(p)

print("differences:")
print("p - f:")
print(p - f)

r = p / f
rNP = r.numpy()
print("frequency ratios:")
print("p / f:")
print(p / f)

c = (1200 / math.log(2)) * torch.log(p / f)
cNP = c.numpy()
print("cent values:")
print((1200 / math.log(2)) * torch.log(p / f))

pNP = p.numpy()

# print("pitch values as numpy array:")
# print(pNP)

print("")
print("Comparisons: (note discrepancies, see comments below)")
print("")

print("key    f_0           YIN           ratio         cents         tau")
for i in range(88):
    print(str(i).ljust(3), "  ", str(fNP[i]).ljust(10), "  ", str(pNP[i]).ljust(10), "  ", str(rNP[i]).ljust(10),
		"  ", str(cNP[i]).ljust(10), "  ", str(sampleRate/fNP[i]).ljust(10))

print("")
print("Note: comparison of piano key frequency f_0 in column 2 and prediction by YIN in column 3")
print("can be seen more precisely by computing the frequency ratio = YIN / f_0 in column 4.")
print("Further, the cent value in column 5 shows that larger discrepancies occur when the value")
print("of tau in column 6 is both small and is close to the midpoint between two integers.")
print("The most extreme value occurs for the last key of the piano, index 87 (key 88):")
print("87     4186.009      4363.6367     1.0424337     71.946815     11.466769999484425")
print("which has value tau about 11.47.  Since the YIN algorithm uses integer values of tau")
print("and frequency uses 1/tau, the largest errors in frequency prediction correspond to")
print("the largest relative error in tau.")

