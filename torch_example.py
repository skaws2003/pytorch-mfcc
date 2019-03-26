import librosa
import torch
import pytorch_mfcc
import numpy


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')     # Device
files = ['english.wav','english_crop.wav']      # Files to load

# Read files
signals = []
wav_lengths = []
sample_rate = 8000  # 8000 for the example file, but normally it is 22050 of 44100. Be careful.

for f in files:
    signal,rate = librosa.load(f,sr=sample_rate,mono=True)    # Load wavefile. Be careful of the sampling rate.
    signals.append(signal)
    wav_lengths.append(len(signal))

# Pad signals with zeros, and make batch
max_length = max(wav_lengths)
signals_torch = []
for i in range(len(signals)):
    signal = torch.tensor(signals[i],dtype=torch.float32).to(device)
    zeros = torch.zeros(max_length - len(signal)).to(device)
    signal = torch.cat([signal,zeros])
    signals_torch.append(signal)
    
signal_batch = torch.stack(signals_torch)

# Now do mfcc
mfcc = pytorch_mfcc.MFCC(samplerate=sample_rate).to(device)
val,mfcc_lengths = mfcc(signal_batch,wav_lengths)

print(val.shape)
print(mfcc_lengths)
