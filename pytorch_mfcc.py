import torch
import torch_dct
import decimal
import numpy

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

class MFCC(torch.nn.Module):
    def __init__(self,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,
                    nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True):
        super(MFCC,self).__init__()
        self.samplerate = samplerate
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft or self.calculate_nfft(self.samplerate,self.winlen)
        self.lowfreq = lowfreq
        self.highfreq = highfreq or self.samplerate/2
        self.preemph = preemph
        self.ceplifter = ceplifter
        self.appendEnergy = appendEnergy
        self.winfunc=lambda x:numpy.ones((x,))


    def calculate_nfft(samplerate, winlen):
        """Calculates the FFT size as a power of two greater than or equal to
        the number of samples in a single window length.
        
        Having an FFT less than the window length loses precision by dropping
        many of the samples; a longer FFT than the window allows zero-padding
        of the FFT buffer which is neutral in terms of frequency domain conversion.

        :param samplerate: The sample rate of the signal we are working with, in Hz.
        :param winlen: The length of the analysis window in seconds.
        """
        window_length_samples = winlen * samplerate
        nfft = 1
        while nfft < window_length_samples:
            nfft *= 2
        return nfft

    
    def forward(self,signal):
        feat,energy = fbank(signal)
        feat = torch.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
        feat = lifter(feat)
        if appendEnergy: feat[:,0] = torch.log(energy) # replace first cepstral coefficient with log of frame energy
        return feat


    def fbank(self,signal):
        """Compute Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The second return value is the energy in each frame (total energy, unwindowed)
        """
        signal = self.preemphasis(signal)
        frames = self.framesig(signal)
        pspec = self.powspec(frames)
        energy = numpy.sum(pspec,1) # this stores the total energy in each frame
        energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

        fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
        feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
        feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

        return feat,energy

    
    def preemphasis(signal, coeff=0.95):
        """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
        :returns: the filtered signal.
        """
        return torch.cat((signal[0].view(1),signal[1:] - self.preemph * signal[:-1]))


    def framesig(signal):
        """Frame a signal into overlapping frames.

        :param sig: the audio signal to frame.
        :returns: an array of frames. Size is NUMFRAMES by frame_len.
        """
        frame_len = self.winlen * self.samplerate
        frame_step = self.winstep * self.samplerate

        slen = len(signal)
        frame_len = int(round_half_up(frame_len))
        frame_step = int(round_half_up(frame_step))
        if slen <= frame_len:
            numframes = 1
        else:
            numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

        padlen = int((numframes - 1) * frame_step + frame_len)

        zeros = torch.zeros((padlen-slen))

        padsignal = torch.cat((sig,zeros))
        
        indices = numpy.tile(numpy.arange(0, self.frame_len), (numframes, 1)) + numpy.tile(numpy.arange(0, numframes * self.frame_step, self.frame_step), (self.frame_len, 1)).T
        ind_shape = indices.shape
        indices = numpy.array(indices, dtype=numpy.int32).reshahpe([-1])
        frames = padsignal[indices].view(ind_shape)
        win = numpy.tile(winfunc(self.frame_len), (numframes, 1))
        win = torch.tensor(win,dtype=frames.dtype).to(frames.device)

        return frames * win

    def powspec(frames):
        """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

        :param frames: the array of frames. Each row is a frame.
        :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
        """
        maged = self.magspec(frames)
        return 1.0 / self.nfft * torch.mul(maged,maged)