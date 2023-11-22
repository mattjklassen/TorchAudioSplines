
# ----- Brief Description -----
# 
# import and export knot sequences to text files knots-[description].txt
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import torch

# this function imports bcoeffs from text file of strings (printed floats)
# and returns a pytorch tensor of those floats.
def import_knots(file) :
    knots_str = []
    with open(file, 'r') as f:
        knots_str = f.readlines()
        f.close()
    N = len(knots_str) - 1
    knots = torch.zeros(N+1)
    for i in range(N+1) :
        knots[i] = float(knots_str[i])
    return knots


def export_knots(file, knots) :
    knots_str = []
    for i in range(len(knots)) :
        knots_str.append(str(float(knots[i])))
        knots_str.append('\n')
    with open(file, 'w') as f:
        f.writelines(knots_str)
        f.close()

