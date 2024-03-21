
import torch
import torchaudio
import numpy as np
import math

from computeBsplineVal import newBsplineVal 
from computeBsplineVal import computeSplineVal 
from genCycle import genCycle
from getBcoeffs import import_bcoeffs, export_bcoeffs

bcoeffs = torch.zeros(3,4)
# bcoeffs[0,0] = 0 already true
bcoeffs[0,1] = 1
bcoeffs[0,2] = 2
bcoeffs[0,3] = 3
bcoeffs[1,0] = 4
bcoeffs[1,1] = 5
bcoeffs[1,2] = 6
bcoeffs[1,3] = 7
bcoeffs[2,0] = 8
bcoeffs[2,1] = 9
bcoeffs[2,2] = 10
bcoeffs[2,3] = 12

def changerow(matrix, row, col, val) :
    matrix[row, col] = val

print("bcoeffs:  ")
print(bcoeffs)

changerow(bcoeffs, 2, 3, 13)

print("modified bcoeffs:  ")
print(bcoeffs)


r0 = bcoeffs[0]
print(r0)

c0 = bcoeffs[:,0]
print(c0)
c1 = bcoeffs[:,1]
print(c1)
c2 = bcoeffs[:,2]
print(c2)

new = torch.zeros(3,4)
new[:,0] = c0
print(new)

for i in range(3) :
    new[:,i+1] = c0 + i+1

print(new)

catrows = torch.cat((c0,c1,c2),0)
print("concat rows along dim=0:")
print(catrows)

catcols = torch.stack((c0,c1,c2),0)
print("stacking cols along dim=0:")
print(catcols)

print("size of c0:", c0.size())
print("int of size of c0:", c0.size(dim=0))

file = "bcoeffs1.txt"
bcoeffs = import_bcoeffs(file)
n = bcoeffs.size(dim=0)
print("size of bcoeffs: ", n)


def change_tensor(input) :
    row = torch.zeros(4)
    for j in range(4) :
        row[j] = 100 + j
    input[1] = row

change_tensor(new)

print(new)

print("row length of new by new.size(dim=1):", new.size(dim=1))
         
print("new[1].size(dim=0): ", new[1].size(dim=0))
print("len(new[1]): ", len(new[1]))

tensor1 = torch.empty(1,3)
tensor2 = torch.zeros(3)
for i in range(3) :
    tensor1[0,i] = tensor2[i]
tensor3 = torch.unsqueeze(tensor2, dim=0)

print("tensor1: ", tensor1)
print("tensor2: ", tensor2)
print("tensor3: ", tensor3)

for i in range(2) :
    print("i value: ", i)
for i in range(1) :
    print("i value: ", i)
for i in range(0) :
    print("i value: ", i)

time1 = 0.125
keys = torch.tensor([0,30,90])
num_keys = keys.size(dim=0)
print("keys: ", keys)
temp = time1 * keys
print("temp keys: ", temp)
for i in range(num_keys) :
    keys[i] = int(temp[i])
print("keys: ", keys)

f0 = 110

for i in range(2001) :
    j = i - 1000
    x = float(j) / 1000.0
    y = np.exp2(x)
    temp_keys = y * temp
    for k in range(num_keys) :
        keys[k] = int(temp_keys[k])
    f = f0 * y
    num_cycles = int(f * 0.125)
#   print(j, " : ", temp_keys, keys, num_cycles)

print("new tensor with dims 3,1,2,4:")
new = torch.zeros(3,1,2,4)
print(new)
new2 = torch.squeeze(new)
print("new squeezed:")
print(new2)





