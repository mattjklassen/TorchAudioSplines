# ----- Brief Description -----
# 
# Record mono from mic at given sample RATE in chunks of size CHUNK samples.
# Duration is in SECONDS, output is output.wav
#
# ----- ----- ----- ----- -----

# ------- More Details --------
# 
#
# ----- ----- ----- ----- -----

import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SECONDS = 2.5

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
		channels=CHANNELS,
		rate=RATE,
		input=True,
		frames_per_buffer=CHUNK)

print("start recording ...")

frames = []
num_chunks = int((RATE * SECONDS) / CHUNK)
num_samples = num_chunks * CHUNK
for i in range(num_chunks + 1):
    data = stream.read(CHUNK)
    frames.append(data)
print("number of chunks: ", num_chunks)
print("number of samples: ", num_samples)

print("recording stopped")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open("output.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


