from struct import pack
from math import sin, pi
import wave

sampling_rate = 44100

# Set up wave file to be generated
wav_file = wave.open('sounds/StereoSoundLevel5.wav', 'w')
wav_file.setparams((2, 2, sampling_rate, 0, 'NONE', 'not compressed'))
max_volume = 2 ** 15 - 1.0  # maximum amplitude
wav_data = b''

# Write data to file
for i in range(0, sampling_rate * 3):
    wav_data += pack('h', round(max_volume * sin(i * 2 * pi * 1000.0 / sampling_rate)))  # 0Hz left
    wav_data += pack('h', round(max_volume * sin(i * 2 * pi * 1000.0 / sampling_rate)))  # 500Hz right
wav_file.writeframes(wav_data)
wav_file.close()
