from lab1_proto import *
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


#############
# Load data #
#############

example = np.load('data/lab1_example.npz', allow_pickle=True)['example'].item()
data = np.load('data/lab1_data.npz', allow_pickle=True)['data']


###############
# 4.1 Enframe #
###############

samples = example['samples']
winlen = int(example["samplingrate"]*20/1000)       # number of samples in 20 ms
winshift = int(example["samplingrate"]*10/1000)     # number of samples in 10 ms
frames = enframe(samples, winlen, winshift)

plt.figure()
plt.pcolormesh(frames)
plt.title('Enframe - computed')
plt.figure()
plt.pcolormesh(example['frames'])
plt.title('Enframe - example')
plt.show()


####################
# 4.2 Pre-emphasis #
####################

preempcoeff = .97
preemph = preemp(frames, preempcoeff)

plt.figure()
plt.pcolormesh(preemph)
plt.title('Pre-emphasis - computed')
plt.figure()
plt.pcolormesh(example['preemph'])
plt.title('Pre-emphasis - example')
plt.show()


######################
# 4.3 Hamming window #
######################

windowed = windowing(preemph)

plt.figure()
plt.pcolormesh(windowed)
plt.title('Hamming window - computed')
plt.figure()
plt.pcolormesh(example['windowed'])
plt.title('Hamming window - example')
plt.show()


###########
# 4.4 FFT #
###########

nfft = 512
spec = powerSpectrum(windowed, nfft)

plt.figure()
plt.pcolormesh(spec)
plt.title('abs(FFT)^2 - computed')
plt.figure()
plt.pcolormesh(example['spec'])
plt.title('abs(FFT)^2 - computed')
plt.show()


######################
# 4.5 Mel filterbank #
######################

samplingrate = example['samplingrate']
mspec_ = logMelSpectrum(spec, samplingrate)

plt.figure()
plt.pcolormesh(mspec_)
plt.title('Mel filterbank - computed')
plt.figure()
plt.pcolormesh(example['mspec'])
plt.title('Mel filterbank - example')
plt.show()


###########
# 4.6 DCT #
###########

nceps = 13
mfcc_ = cepstrum(mspec_, nceps)
lmfcc = lifter(mfcc_)

plt.figure()
plt.pcolormesh(mfcc_)
plt.title('MFCC - computed')
plt.figure()
plt.pcolormesh(example['mfcc'])
plt.title('MFCC - example')
plt.show()

plt.figure()
plt.pcolormesh(lmfcc)
plt.title('Liftered MFCC - computed')
plt.figure()
plt.pcolormesh(example['lmfcc'])
plt.title('Liftered MFCC - example')
plt.show()

#########################
# 5 Feature correlation #
#########################

# MFCCs and mspec of all utterances
for utterance in data:
    samples = utterance['samples']
    utterance['mfcc'] = mfcc(samples, winlen, winshift, preempcoeff, nfft, nceps, samplingrate)
    utterance['mspec'] = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)

# show some utterances
# for utterance in data[[0, 4]]:
#     plt.figure()
#     plt.pcolormesh(utterance['mfcc'])
#     plt.title('Gender: {}, Digit: {}, Repetition: {}'.format(utterance['gender'], utterance['digit'], utterance['repetition']))
# plt.show()

# concatenate frames
mfcc_frames = np.concatenate([u['mfcc'] for u in data], axis=0)
mspec_frames = np.concatenate([u['mspec'] for u in data], axis=0)

# correlations of features
correlations_mfcc = np.corrcoef(mfcc_frames, rowvar=False)
correlations_mspec = np.corrcoef(mspec_frames, rowvar=False)

plt.figure()
plt.pcolormesh(correlations_mfcc)
plt.title('Correlation matrix - MFCC')
plt.show()

plt.figure()
plt.pcolormesh(correlations_mspec)
plt.title('Correlation matrix - Mel filterbank')
plt.show()


#############################################
# 6 Explore speech segments with clustering #
#############################################

np.random.seed(1)

# train GMM
gmm = GaussianMixture(n_components=32, covariance_type='diag', n_init=100)
gmm.fit(mfcc_frames)

# analyze some utterances
for utterance in data[[16, 17, 38, 39]]:
    # plot posteriors
    plt.figure()
    plt.pcolormesh(gmm.predict_proba(utterance['mfcc']))
    plt.title('Gender: {}, Digit: {}, Repetition: {}'.format(utterance['gender'], utterance['digit'], utterance['repetition']))
    plt.xlabel('component')
    plt.ylabel('frame')
plt.show()


##########################
# 7 Comparing utterances #
##########################

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


N = len(data)   # number of utterances
D = np.zeros((N, N))

# compute distance matrix
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        x = data[i]['mfcc']
        y = data[j]['mfcc']
        D[i, j] = dtw(x, y, euclidean_distance)[0]

# display distance matrix
plt.figure()
plt.pcolormesh(D)
plt.title('Distance matrix')
plt.show()

# hierarchical clustering
Z = linkage(squareform(D), method='complete')
plt.figure()
dendrogram(Z, labels=tidigit2labels(data))
plt.show()
