

# ----- Brief Description -----
# 
# This is a batch script to do generation of polyphonic tones. 
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import subprocess


numstr = "4-5-6-7"
subprocess.run(["python", "gen2PolyWaveform.py", "bcoeffs.txt", "4", "5", "6", "7"])

bcoeffs = "poly_tones/" + numstr + "/poly_bcoeffs.txt"
subprocess.run(["python", "plotPdfBcoeffs.py", "poly_tones/4-5-6-7/poly_bcoeffs.txt"])

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


