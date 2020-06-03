import matplotlib.pyplot as plt
import warnings
import timeit
from lab2_proto import *
from prondict import prondict


def pcolormesh(matrix, title):
    plt.pcolormesh(matrix.T)
    plt.title(title)
    plt.ylabel('state')
    plt.xlabel('time step')
    plt.yticks(np.arange(matrix.shape[1]) + .5, range(matrix.shape[1]))
    plt.xticks(np.arange(0, matrix.shape[0], 10) + .5, range(0, matrix.shape[0], 10))


warnings.filterwarnings(action='ignore', category=RuntimeWarning)


###################
# Data and models #
###################

# data
example = np.load('data/lab2_example.npz', allow_pickle=True)['example'].item()
data = np.load('data/lab2_data.npz', allow_pickle=True)['data']

# phonetic models (HMMs for phonemes)
phoneHMMs_onespkr = np.load('data/lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
phoneHMMs_all = np.load('data/lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

# lexical model (isolated word recognition => initial and final silence)
isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']

# word models (HMMs for words)
wordHMMs_onespkr = {}
wordHMMs_all = {}
for digit in prondict.keys():
    wordHMMs_onespkr[digit] = concatHMMs(phoneHMMs_onespkr, isolated[digit])
    wordHMMs_onespkr[digit]['name'] = digit

    wordHMMs_all[digit] = concatHMMs(phoneHMMs_all, isolated[digit])
    wordHMMs_all[digit]['name'] = digit


##############################
# 5.1 Emission probabilities #
##############################

wordHMM = wordHMMs_onespkr[example['digit']]
obsloglik = log_multivariate_normal_density_diag(example['lmfcc'], wordHMM['means'], wordHMM['covars'])

plt.figure()
pcolormesh(obsloglik, 'Emission probabilities - computed')
plt.figure()
pcolormesh(example['obsloglik'], 'Emission probabilities - example')


#########################
# 5.2 Forward algorithm #
#########################

def score_recognizer(wordHMMs, data, viterbi_approx=False):
    accuracy = 0

    for utterance in data:
        best_loglik = np.NINF

        for word, wordHMM in wordHMMs.items():
            obsloglik = log_multivariate_normal_density_diag(utterance['lmfcc'], wordHMM['means'], wordHMM['covars'])

            if viterbi_approx:
                loglik = viterbi(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))[0]
            else:
                loglik = forward(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))[1]

            if loglik > best_loglik:
                best_loglik = loglik
                recognized_word = word

        if utterance['digit'] == recognized_word:
            accuracy += 1

    accuracy /= len(data)
    return accuracy


logalpha, loglik = forward(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))

plt.figure()
pcolormesh(logalpha, r'Log $\alpha$ - computed')
plt.figure()
pcolormesh(example['logalpha'], r'Log $\alpha$ - example')

print('Log-likelihood, computed (forward):', loglik)
print('Log-likelihood, example:', example['loglik'])
print()

print('Accuracy, training on one speaker: {:.1f}%'.format(100 * score_recognizer(wordHMMs_onespkr, data)))
print('Accuracy, training on all speakers: {:.1f}%'.format(100 * score_recognizer(wordHMMs_all, data)))
print()


#############################
# 5.3 Viterbi approximation #
#############################

vloglik, vpath = viterbi(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))

print('Viterbi log-likelihood, computed:', vloglik)
print('Viterbi log-likelihood, example:', example['vloglik'])
print()

print('Accuracy with Viterbi approximation, training on one speaker: {:.1f}%'
      .format(100 * score_recognizer(wordHMMs_onespkr, data, viterbi_approx=True)))
print('Accuracy with Viterbi approximation, training on all speakers: {:.1f}%'
      .format(100 * score_recognizer(wordHMMs_all, data, viterbi_approx=True)))
print()

# alphas overlaid by best best
plt.figure()
pcolormesh(logalpha, r'Log $\alpha$ with best path')
plt.plot(np.arange(len(vpath)) + .5, example['vpath'] + .5, 'k', linewidth=2, label='example')
plt.plot(np.arange(len(vpath)) + .5, vpath + .5, 'r--', linewidth=1, label='computed')
plt.legend()

# efficiency forward vs Viterbi
t_forward = timeit.default_timer()
score_recognizer(wordHMMs_onespkr, data)
score_recognizer(wordHMMs_all, data)
t_forward = timeit.default_timer() - t_forward

t_viterbi = timeit.default_timer()
score_recognizer(wordHMMs_onespkr, data, viterbi_approx=True)
score_recognizer(wordHMMs_all, data, viterbi_approx=True)
t_viterbi = timeit.default_timer() - t_viterbi

print('Time elapsed with forward: {:.2f} s'.format(t_forward))
print('Time elapsed with Viterbi: {:.2f} s'.format(t_viterbi))
print()


##########################
# 5.4 Backward algorithm #
##########################

logbeta, loglik = backward(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))

plt.figure()
pcolormesh(logbeta, r'Log $\beta$ - computed')
plt.figure()
pcolormesh(example['logbeta'], r'Log $\beta$ - example')

print('Log-likelihood, computed (backward):', loglik)
print('Log-likelihood, example:', example['loglik'])
print()


########################
# 6.1 State posteriors #
########################

loggamma = statePosteriors(logalpha, logbeta)

plt.figure()
pcolormesh(loggamma, r'Log $\gamma$ - computed')
plt.figure()
pcolormesh(example['loggamma'], r'Log $\gamma$ - example')

# row-stochastic
if not all(np.isclose(np.exp(logsumexp(loggamma, axis=1)), 1)):
    raise ValueError('log_gamma is not row-stochastic.')

# GMMs vs HMMs
loggamma_gmm = obsloglik - logsumexp(obsloglik, axis=1).reshape(-1, 1)
plt.figure()
pcolormesh(loggamma_gmm, r'Log $\gamma^{GMM}$')

# sum of posteriors along time axis
print('Sum of posteriors along time axis:\n', np.exp(logsumexp(loggamma)))
print()

# sum of all posteriors vs length of observation sequence
print('Sum of all posteriors: {:.2f}'.format(np.exp(logsumexp(logsumexp(loggamma)))))
print('Length of observation sequence: {}'.format(len(obsloglik)))
print()


#########################################
# 6.2 Retraining emission distributions #
#########################################

def retrain_emission_distributions(utterance, wordHMM, maxiter=20, threshold=1.0):
    loglik = []

    for i in range(maxiter):
        # E-step
        obsloglik = log_multivariate_normal_density_diag(utterance['lmfcc'], wordHMM['means'], wordHMM['covars'])
        logalpha, loglik_ = forward(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))
        logbeta = backward(obsloglik, np.log(wordHMM['startprob']), np.log(wordHMM['transmat']))[0]
        loggamma = statePosteriors(logalpha, logbeta)

        # check termination (here to avoid running forward() two times per iteration)
        if i > 0 and abs(loglik_ - loglik[-1]) < threshold:
            break
        loglik.append(loglik_)

        # M-step
        wordHMM['means'], wordHMM['covars'] = updateMeanAndVar(utterance['lmfcc'], loggamma)

    return loglik


plt.figure()
utterance = data[10]
for word, wordHMM in wordHMMs_all.items():
    loglik = retrain_emission_distributions(utterance, wordHMM)

    plt.plot(loglik, label='HMM for digit {}'.format(word))
    plt.xlabel('# iterations')
    plt.ylabel('log-likelihood')
    plt.title("Learning curve (only emission distributions, utterance of '{}')".format(utterance['digit']))
plt.legend()
plt.show()
