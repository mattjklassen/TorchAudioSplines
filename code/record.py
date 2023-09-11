import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 44100
RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
		channels=CHANNELS,
		rate=RATE,
		input=True,
		frames_per_buffer=CHUNK)

print("start recording ...")

frames = []
seconds = 1
# changing to read 10 chunks, about 2/3 sec
# 10240 samples at rate f_s=16000 is 0.64 sec
# (hard coded) for i in range(0, 9):
# 1 sec of audio at RATE=f_s==44100 is about 43 chunks of size 1024
# ie. int(RATE / CHUNK) = 43
i = 0
for i in range(0, int(RATE / CHUNK * seconds)+1):
    data = stream.read(CHUNK)
    frames.append(data)
print("number of chunks: ", i)
    
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


