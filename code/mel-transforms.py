

# ----- Brief Description -----
# 
# This is a batch script to do melody generation with several parameters reset in a loop.
# We use the python subprocess module to run command line programs like mel.py with config.txt
# We use parameters on the command line, which can override the settings in mel.py
# such as "invert=1" to do inversion of melody
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import subprocess

y = "0.1"
# name = "fluteA440"
# name = "frhorn315"
name = "cello189"

source = name + "/"
target = name + "/mel-s" + y + "/"

subprocess.run(["python", "mel.py"])
subprocess.run(["python", "mel.py", "invert=1"])
subprocess.run(["python", "mel.py", "retro=1"])
subprocess.run(["python", "mel.py", "invert=1", "retro=1"])

subprocess.run(["cp", source + "melody-prime.wav", target + name + "-mel-p-" + y + ".wav"])
subprocess.run(["cp", source + "melody-inversion.wav", target + name + "-mel-i-" + y + ".wav"])
subprocess.run(["cp", source + "melody-retrograde.wav", target + name + "-mel-r-" + y + ".wav"])
subprocess.run(["cp", source + "melody-retrograde-inversion.wav", target + name + "-mel-ri-" + y + ".wav"])

subprocess.run(["cp", source + "prime-melody-summary.txt", target])
subprocess.run(["cp", source + "inversion-melody-summary.txt", target])
subprocess.run(["cp", source + "retrograde-melody-summary.txt", target])
subprocess.run(["cp", source + "retrograde-inversion-melody-summary.txt", target])


