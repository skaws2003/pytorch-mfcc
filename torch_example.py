import scipy.io.wavfile as wav
import torch
import pytorch_mfcc

(rate,sig) = wav.read("english.wav")
signal = torch.tensor(sig,dtype=torch.float32).cuda()

mfcc = pytorch_mfcc.MFCC(samplerate=rate).cuda()
val = mfcc(signal)

print(val.shape)
print(val[44][3])
