# Now official torchaudio supports MFCC!!! See [Here](https://github.com/pytorch/audio). This Library will no longer be maintained

# MFCC (Mel Frequency Cepstral Coefficient) for PyTorch

Based on [this repository](https://github.com/jameslyons/python_speech_features), this project extends the MFCC function for Pytorch so that backpropagation path could be established through.


## Dependency
* Python >= 3.5
* PyTorch >= 1.0
* numpy
* librosa


## Installation
```
git clone https://github.com/skaws2003/pytorch_mfcc.git
```

## Parameters
| Parameters   	|     Description                                                                                    	|
|--------------	|------------------------------------------------------------------------------------------------------	|
| samplerate   	| samplerate of the signal                                                                             	|
|  winlen      	| the length of the analysis window. Defaults 0.025s                                                   	|
| winstep      	| the length of step between each windows. Defaults 0.01s                                              	|
| numcep       	| the number of cepstrum to return. Defaults 13                                                        	|
| nfilt        	| the number of filters in the filterbank. Defaults 26                                                 	|
| nfft         	| FFT size. Defaults 512                                                                               	|
| lowfreq      	| lowest band edge of mel filters(Hz) Defaults 0                                                       	|
| highfreq     	| highest band edge of mel filters(Hz) Defaults samplerate/2                                           	|
| preemph      	| apply preemphasis filter with preemph as coefficient. 0 is no filter. Defaults 0.97                  	|
| ceplifter    	| apply a lifter to final cepstral coefficients. 0 is no lifter. Defaults 22                           	|
| appendEnergy 	| if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy. 	|




## Example use
```python
import librosa
import torch
import pytorch_mfcc
import numpy


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')     # Device
files = ['english.wav','english_crop.wav']      # Files to load

# Read files
signals = []
wav_lengths = []
sample_rate = 8000  # 8000 for the example file, but normally it is 22050 of 44100. Check it and be careful.

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
mfcc_layer = pytorch_mfcc.MFCC(samplerate=sample_rate).to(device)     # MFCC layer
val,mfcc_lengths = mfcc_layer(signal_batch,wav_lengths)       # Do mfcc

print(val.shape)
print(mfcc_lengths)
```

## References
* [DCT for PyTorch](https://github.com/zh217/torch-dct) by [Ziyang Hu](https://github.com/zh217/)
* This project is based on [python_speech_features](https://github.com/jameslyons/python_speech_features) by [James Lyons](https://github.com/jameslyons)


## Sample Source
sample english.wav and english_crop.wav from:
```
wget http://voyager.jpl.nasa.gov/spacecraft/audio/english.au
sox english.au -e signed-integer english.wav
```


## Comments
Any contribution is welcomed. Please don't hesitate to make a pull request.
