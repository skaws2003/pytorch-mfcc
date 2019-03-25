import scipy.io.wavfile as wav
import torch
import pytorch_mfcc

(rate,sig) = wav.read("english.wav")
signal = torch.tensor(sig,dtype=torch.float32)

mfcc = pytorch_mfcc.MFCC(samplerate=rate)
val = mfcc(signal)

print(val.shape)
print(val[44][3])
