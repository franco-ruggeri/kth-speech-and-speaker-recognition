import warnings
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Accuracy
from edit_distance import SequenceMatcher
from lab3_proto import *
from lab1_proto import mfcc, mspec
from lab2_proto import concatHMMs, viterbi
from lab2_tools import log_multivariate_normal_density_diag
from prondict import prondict


def pcolormesh(matrix, title, ylabel):
    plt.pcolormesh(matrix.T)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('time step')
    plt.yticks(np.arange(matrix.shape[1]) + .5, range(matrix.shape[1]))
    plt.xticks(np.arange(0, matrix.shape[0], 10) + .5, range(0, matrix.shape[0], 10))


DEBUG = False
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
np.random.seed(1)


###############################
# 4.1 Target class definition #
###############################

# unique states (=> classes for DNN)
phoneHMMs = np.load('data/lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [p + '_' + str(i) for p in phones for i in range(nstates[p])]


########################
# 4.2 Forced alignment #
########################

if DEBUG:
    # load correct example
    example = np.load('data/lab3_example.npz', allow_pickle=True)['example'].item()

    # feature extraction
    filename = 'data/tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)

    # transcription
    wordTrans = list(path2info(filename)[2])            # word transcription (contained in the filename)
    phoneTrans = words2phones(wordTrans, prondict)      # word transcription => phone transcription
    stateTrans = [p + '_' + str(i) for p in phoneTrans
                  for i in range(nstates[p])]           # phone transcription => state transcription

    # combined HMM for utterance
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

    # Viterbi decoder
    obsloglik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
    viterbiLoglik, viterbiPath = viterbi(obsloglik, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat']))

    # time alignment (frame-by-frame state transcription)
    viterbiStateTrans = [stateTrans[s] for s in viterbiPath]

    # save in standard format (to use it, put it in the same directory of .wav and open .wav with wavesurfer)
    frames2trans(viterbiStateTrans, outfilename='data/transcriptions/z43a.lab')

    # check results
    plt.figure()
    pcolormesh(lmfcc, 'MFCC - computed', ylabel='MFCC')
    plt.figure()
    pcolormesh(example['lmfcc'], 'MFCC - example', ylabel='MFCC')

    plt.figure()
    pcolormesh(obsloglik, 'Emission probabilities - computed', ylabel='state')
    plt.figure()
    pcolormesh(example['obsloglik'], 'Emission probabilities - example', ylabel='state')

    plt.figure()
    plt.plot(np.arange(len(viterbiPath)) + .5, example['viterbiPath'] + .5, 'k', linewidth=2, label='example')
    plt.plot(np.arange(len(viterbiPath)) + .5, viterbiPath + .5, 'r--', linewidth=1, label='computed')
    plt.legend()
    plt.title('Viterbi path')
    plt.show()

    print('Word transcription, computed:', wordTrans)
    print('Word transcription, example:', example['wordTrans'])
    print()

    print('Phone transcription, computed:', phoneTrans)
    print('Phone transcription, example:', example['phoneTrans'])
    print()

    print('State transcription, computed:', stateTrans)
    print('State transcription, example:', example['stateTrans'])
    print()

    print('Viterbi log-likelihood, computed:', viterbiLoglik)
    print('Viterbi log-likelihood, example:', example['viterbiLoglik'])
    print()

    print('Forced alignment, computed:', viterbiStateTrans)
    print('Forced alignment, example:', example['viterbiStateTrans'])
    print()


##########################
# 4.3 Feature extraction #
##########################

def extract_feature(path, states):
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                print(filename + '... ', end='')

                # feature extraction (=> inputs for DNN)
                lmfcc = mfcc(samples)
                mspec_ = mspec(samples)

                # forced alignment (=> targets for DNN)
                wordTrans = list(path2info(filename)[2])                # word transcription (contained in the filename)
                phoneTrans = words2phones(wordTrans, prondict)          # word transcription => phone transcription
                targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
                targets = np.array([states.index(t) for t in targets])  # save targets as indeces

                data.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec_, 'targets': targets})
                print('done')
    return np.array(data)


# training set
filename = 'data/traindata.npz'
if os.path.isfile(filename):
    print('loading training set... ', end='')
    traindata = np.load(filename, allow_pickle=True)['traindata']
    print('done')
else:
    traindata = extract_feature('data/tidigits/disc_4.1.1/tidigits/train', stateList)
    np.savez(filename, traindata=traindata)

# test set
filename = 'data/testdata.npz'
if os.path.isfile(filename):
    print('loading test set... ', end='')
    testdata = np.load(filename, allow_pickle=True)['testdata']
    print('done')
else:
    testdata = extract_feature('data/tidigits/disc_4.2.1/tidigits/test', stateList)
    np.savez(filename, testdata=testdata)


####################################
# 4.4 Training and validation sets #
####################################

def split_data_gender(data, percentage):
    n = len(data)
    speakers = list({path2info(u['filename'])[1] for u in data})    # set => no duplicates!
    random.shuffle(speakers)                                        # shuffle to randomly select speakers
    val_data = []

    # add to validation set (all utterances of) speakers until the percentage is reached
    for s in speakers:
        speaker_data = [u for u in data if path2info(u['filename'])[1] == s]
        val_data.extend(speaker_data)

        # check if percentage is reached
        if len(val_data) > percentage * n:
            break

    # add rest to training set
    train_data = [u for u in data if u not in val_data]

    return train_data, val_data


def split_data(data, percentage):
    train_data = []
    val_data = []

    # stratified sampling:
    # 1. split data by gender
    # 2. sample from these partitions preserving the distribution

    # split data by gender
    men_data = [u for u in data if path2info(u['filename'])[0] == 'man']
    women_data = [u for u in data if path2info(u['filename'])[0] == 'woman']

    # add utterances of men
    train_data_, val_data_ = split_data_gender(men_data, percentage)
    train_data.extend(train_data_)
    val_data.extend(val_data_)

    # add utterances of women
    train_data_, val_data_ = split_data_gender(women_data, percentage)
    train_data.extend(train_data_)
    val_data.extend(val_data_)

    return np.array(train_data), np.array(val_data)


print('extracting validation set... ', end='')
traindata, valdata = split_data(traindata, .1)
print('done')


###########################################
# 4.5 Acoustic context (dynamic features) #
###########################################

def stack_acoustic_context(features):
    length = np.shape(features)[0]
    idx_list = list(range(length))
    idx_list = idx_list[1:4][::-1] + idx_list + idx_list[-4:-1][::-1]
    output = [features[idx_list[i:i + 7]].reshape(-1) for i in range(length)]
    return np.array(output)


###############################
# 4.6 Feature standardisation #
###############################

def prepare_matrices(data, K, feature_type, dynamic_features, scaler=None):
    if feature_type != 'lmfcc' and feature_type != 'mspec':
        raise ValueError('Invalid feature type. Choose among lmfcc and mspec.')

    N = sum([len(u['targets']) for u in data])      # total number of frames
    D = data[0][feature_type].shape[1]
    if dynamic_features:
        D = D*7

    x = np.zeros((N, D), dtype='float32')           # float32 to save memory
    y = np.zeros(N)

    # flatten (i.e. concatenate frames and targets of all utterances)
    i = 0
    for u in data:
        n = len(u['targets'])                                       # number of frames for utterance
        if dynamic_features:
            u_features = stack_acoustic_context(u[feature_type])    # add dynamic features
        else:
            u_features = u[feature_type]

        x[i:i+n] = u_features
        y[i:i+n] = u['targets']
        i += n

    # standardise
    if scaler is not None:
        x = scaler.transform(x)

    # one-hot encoding
    y = to_categorical(y, K)

    return x, y


feature_type = 'lmfcc'      # select here features to use
dynamic_features = True     # select here whether to use dynamic features

K = len(stateList)          # number of classes (don't touch)
print('preparing matrices... ', end='')

# matrices for training set
train_x, train_y = prepare_matrices(traindata, K, feature_type, dynamic_features=dynamic_features)

# standardisation of training set
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)

# matrices for validation and test sets (already standardised)
val_x, val_y = prepare_matrices(valdata, K, feature_type, dynamic_features=dynamic_features, scaler=scaler)
test_x, test_y = prepare_matrices(testdata, K, feature_type, dynamic_features=dynamic_features, scaler=scaler)
print('done')


####################################
# 5. Phoneme recognition with DNNs #
####################################

model_dir = 'models'
model_name = 'dnn_4hl_256u_dlmfcc_relu_bn'
model_filepath = os.path.join(model_dir, model_name + '.h5')

if not os.path.isfile(model_filepath):
    # create directory to save model
    try:
        os.mkdir(model_dir)
        print('Directory', model_dir, 'created')
    except FileExistsError:
        print('Directory', model_dir, 'already existing')

    # architecture
    dnn = Sequential()
    for i in range(4):
        if i == 0:
            dnn.add(Dense(units=256, input_shape=(train_x.shape[1],)))
        else:
            dnn.add(Dense(units=256))
        dnn.add(BatchNormalization())
        dnn.add(Activation(activation='relu'))
    dnn.add(Dense(units=K, activation='softmax'))
    dnn.summary()

    # compile
    dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train
    checkpoint = ModelCheckpoint(model_filepath, verbose=1, save_best_only=True)
    history = dnn.fit(x=train_x, y=train_y, batch_size=256, epochs=10, validation_data=(val_x, val_y),
                      callbacks=[checkpoint])

    # plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(model_dir, model_name + '_loss.jpg'))

    # plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(model_dir, model_name + '_accuracy.jpg'))

# load (best checkpoint of) model
print('loading best checkpoint... ', end='')
dnn = load_model(model_filepath)
print('done')
dnn.summary()

# evaluate on validation set (for model selection)
metric_values = dnn.evaluate(x=val_x, y=val_y, batch_size=256)
with open(os.path.join(model_dir, 'model_selection.txt'), 'a') as f:
    f.write(model_name + '\n')
    for n, v in zip(dnn.metrics_names, metric_values):
        print('validation {}: {}'.format(n, v))
        f.write('validation {}: {}\n'.format(n, v))
    f.write('\n\n')


###########################
# 5.1 Detailed evaluation #
###########################

def states2phones(y_states, phones, states):
    # work in one-hot encoding
    y_phones = np.zeros((y_states.shape[0], len(phones)))
    y_states = to_categorical(y_states, len(states))

    for i, p in enumerate(phones):
        # indeces of states belonging to phoneme p
        idx_states = [j for j in range(len(states)) if states[j].split(sep='_')[0] == p]

        # merge states (just sum columns in one-hot encoding!)
        y_phones[:, i] = np.sum(y_states[:, idx_states], axis=1)

    # go back to labels
    y_phones = np.argmax(y_phones, axis=1)

    return y_phones


def merge_consequent_states(y):
    cur_y = y[0]
    merged_y = [cur_y]
    for i in y:
        if i == cur_y:
            continue
        else:
            cur_y = i
            merged_y += [cur_y]
    return merged_y


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.imshow(cm, cmap='Blues')
    plt.xlabel('predictions')
    plt.ylabel('ground truth')


# predict
y_pred = np.argmax(dnn.predict(x=test_x, batch_size=256), axis=1)
y_true = np.argmax(test_y, axis=1)
accuracy = Accuracy()

# frame-by-frame at the state level
accuracy.update_state(y_true, y_pred)
print('Frame-by-frame accuracy at the state level: {:.2f}%'.format(accuracy.result().numpy()*100))
plt.figure()
plot_confusion_matrix(y_true, y_pred)
plt.title('Frame-by-frame confusion matrix at the state level')

# frame-by-frame at the phoneme level
y_pred_phones = states2phones(y_pred, phones, stateList)
y_true_phones = states2phones(y_true, phones, stateList)
accuracy.reset_states()
accuracy.update_state(y_true_phones, y_pred_phones)
print('Frame-by-frame accuracy at the phoneme level: {:.2f}%'.format(accuracy.result().numpy()*100))
plt.figure()
plot_confusion_matrix(y_true_phones, y_pred_phones)
plt.title('Frame-by-frame confusion matrix at the phoneme level')

# PER at the state level
N = 10000     # number of frames to consider (distance computation is expensive)
y_pred_merged = merge_consequent_states(y_pred[:N])
y_true_merged = merge_consequent_states(y_true[:N])
sm = SequenceMatcher(a=y_true_merged, b=y_pred_merged)
edit_distance = sm.distance()
print('PER at the state level: {:.2f}%'.format(edit_distance/N*100))

# PER at the phoneme level
y_pred_merged = merge_consequent_states(y_pred_phones[:N])
y_true_merged = merge_consequent_states(y_true_phones[:N])
sm = SequenceMatcher(a=y_true_merged, b=y_pred_merged)
edit_distance = sm.distance()
print('PER at the phoneme level: {:.2f}%'.format(edit_distance/N*100))

# posteriors for first utterance
utterance = testdata[0]
x, y = prepare_matrices([utterance], K, feature_type, dynamic_features=dynamic_features, scaler=scaler)
y_pred = dnn.predict(x=x, batch_size=256)
wordTrans = path2info(utterance['filename'])[2]
plt.figure()
pcolormesh(y_pred, 'Predicted posteriors (produced by DNN) - words: ' + wordTrans, ylabel='state')
plt.figure()
pcolormesh(y, 'Target posteriors (produced by forced alignment) - words: ' + wordTrans, ylabel='state')
plt.show()
