

# ----- Brief Description -----
# 
# This is a batch script to do melody generation with several parameters reset in a loop.
# We use the python subprocess module to run command line programs like findCycles.py
# We use parameters on the command line such as <audiofilename.wav> <n> <seg_num> <f0_guess>
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import subprocess

source = "frhorn315.wav"

for i in range(35) :
    subprocess.run(["python", "findCycles.py", source, "30", str(i), "315"])


