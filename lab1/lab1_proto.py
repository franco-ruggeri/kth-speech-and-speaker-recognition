# DT2119, Lab 1 Feature Extraction

from lab1_tools import *
import matplotlib.pyplot as plt
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct


# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)


# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    if len(samples) < winlen:
        raise ValueError('Too long winlen wrt input signal.')

    N = 1 + int((len(samples) - winlen) / winshift)     # first window + how many times the window can be shifted
    frames = np.zeros((N, winlen))

    # init window
    start = 0
    end = start + winlen

    for i in range(N):
        # save frame
        frames[i] = samples[start:end]

        # shift window
        start = start + winshift
        end = start + winlen
    return frames


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    # y[n] = x[n] - p*x[n-1]
    num = [1, -p]
    den = [1]
    return lfilter(num, den, input)


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=False option
    if you want to get the same results as in the example)
    """
    w = hamming(input.shape[1], sym=False)

    # window shape (for explanation)
    # plt.figure()
    # plt.plot(w)
    # plt.title('Hamming window')
    # plt.xlabel('sample')
    # plt.ylabel('amplitude')
    # plt.show()

    return input * w


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    return np.absolute(fft(input, n=nfft)) ** 2


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = input.shape[1]
    filterbank = trfbank(samplingrate, nfft)

    # filters (for explanation)
    # plt.figure()
    # plt.plot(filterbank[::5])
    # plt.title('Filterbank')
    # plt.xlabel('frequency')
    # plt.ylabel('amplitude')
    # plt.show()

    return np.log(input @ filterbank.T)


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    cepstrum = dct(input)
    cepstrum = cepstrum[:, :nceps]
    return cepstrum


def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path through AD

    Note that you only need to define the first output for this exercise.
    """
    N = x.shape[0]
    M = y.shape[0]

    # local distance between frames (frame-wise distance)
    LD = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            LD[i, j] = dist(x[i], y[j])

    # accumulated distances
    AD = np.zeros((N, M))
    pred = np.zeros((N, M, 2), dtype='uint8')   # the predecessor is represented as tuple of coordinates
    for i in range(N):
        for j in range(M):
            if i != 0 or j != 0:
                # find minimum and save predecessor
                candidates_pred = []
                ad_candidates_pred = []
                if i > 0:
                    candidates_pred.append((i-1, j))
                    ad_candidates_pred.append(AD[i-1, j])
                if i > 0 and j > 0:
                    candidates_pred.append((i-1, j-1))
                    ad_candidates_pred.append(AD[i-1, j-1])
                if j > 0:
                    candidates_pred.append((i, j-1))
                    ad_candidates_pred.append(AD[i, j-1])
                m = min(ad_candidates_pred)
                idx_candidate = ad_candidates_pred.index(m)
                pred[i, j] = candidates_pred[idx_candidate]
            else:
                m = 0

            # compute accumulated distance
            AD[i, j] = LD[i, j] + m

    # global distance
    d = AD[-1, -1] / (N + M)

    # best path (backtracking)
    path = [(N-1, M-1)]
    current = path[0]
    while current != (0, 0):
        path.insert(0, tuple(pred[current]))
        current = path[0]

    # show best path (debugging)
    # LD_ = LD.copy()
    # for node in path:
    #     LD_[node] = 0   # to be visible in black
    # plt.figure()
    # plt.pcolormesh(LD_)
    # plt.show()

    return d, LD, AD, path
