# MFCC (Mel Frequency Cepstral Coefficient) for PyTorch

Based on [this repository](https://github.com/jameslyons/python_speech_features), this project extends the MFCC function for Pytorch so that backpropagation path could be established through.

This implementation is alpha. Batch support is not yet implemented, but will be soon.

## Dependency
* tested on PyTorch 1.0 (but probably works with >= 0.4)
* [torch-dct](https://github.com/zh217/torch-dct)
* numpy


## Installation
```
pip install torch-dct
git clone https://github.com/skaws2003/pytorch_mfcc.git
```


## Usage
```python
import scipy.io.wavefile as wav
import torch
import pytorch_mfcc

(rate,sig) = wav.read("english.wav")
signal = torch.tensor(sig)

mfcc = pytorch_mfcc.MFCC(samplerate=rate)
val = mfcc(signal)

print(val.shape)

>>> torch.Size([426,13])
```


## Acknowledgements
* [DCT transformation for PyTorch](https://github.com/zh217/torch-dct) by [Ziyang Hu](https://github.com/zh217/)
* This code is based on [python_speech_features](https://github.com/jameslyons/python_speech_features) by [James Lyons](https://github.com/jameslyons)


## Reference
sample english.wav from:
```
wget http://voyager.jpl.nasa.gov/spacecraft/audio/english.au
sox english.au -e signed-integer english.wav
```


## Future works
* Make the functions support batch
* Implement other features of python_speech_features
* distribute over pypi
* sphinx documentation


## Comments
Any contribution is welcomed. Please don't hesitate to make a pull request.
