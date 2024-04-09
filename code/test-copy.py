

# ----- Brief Description -----
# 
# This is a batch script to do melody generation with several parameters reset in a loop.
# We use the python subprocess module to run command line programs like findCycles.py and melody6.py.
# We use parameters on the command line, which can override the settings in melody6.py
# such as "invert=1" to do inversion of melody
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import subprocess

subprocess.run(["cp", "dulcimerA3-f/prime-melody-summary.txt", "dulcimerA3-f/greg/mel6/"])
subprocess.run(["cp", "dulcimerA3-f/inversion-melody-summary.txt", "dulcimerA3-f/greg/mel6/"])
subprocess.run(["cp", "dulcimerA3-f/retrograde-melody-summary.txt", "dulcimerA3-f/greg/mel6/"])
subprocess.run(["cp", "dulcimerA3-f/retrograde-inversion-melody-summary.txt", "dulcimerA3-f/greg/mel6/"])
