# MFCC (Mel Frequency Cepstral Coefficient) for PyTorch

Based on [this repository](https://github.com/jameslyons/python_speech_features), this project extends the MFCC function for Pytorch so that backpropagation path could be established through.


## Dependency
* PyTorch >= 1.0
* numpy


## Installation
```
git clone https://github.com/skaws2003/pytorch_mfcc.git
```


## Acknowledgements
* [DCT for PyTorch](https://github.com/zh217/torch-dct) by [Ziyang Hu](https://github.com/zh217/)
* This code is based on [python_speech_features](https://github.com/jameslyons/python_speech_features) by [James Lyons](https://github.com/jameslyons)


## Reference
sample english.wav from:
```
wget http://voyager.jpl.nasa.gov/spacecraft/audio/english.au
sox english.au -e signed-integer english.wav
```


## Future works
* Optimize with batch working
* Implement other features of python_speech_features
* distribute over pypi
* sphinx documentation


## Comments
Any contribution is welcomed. Please don't hesitate to make a pull request.
