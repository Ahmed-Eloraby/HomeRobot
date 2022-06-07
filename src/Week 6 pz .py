from pprint import pprint

import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

window = 240
latency = 0
channel = 11 - 1  # cz
# channels = np.array([10,33,48,50,52,55,59,61])
channels = np.arange(64)

freq = 240
lowcut_freq = 0.1
highcut_freq = 20

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut=lowcut_freq, highcut=highcut_freq, fs=freq):
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = lfilter(b, a, data)
    # f_signal = rfft(data)

    # W = rfftfreq(data.size, d=1 / freq)
    # cut_f_signal = f_signal.copy()
    # cut_f_signal[(W > highcut_freq)] = 0
    # cut_f_signal[(W < lowcut_freq)] = 0
    #
    # cut_signal = irfft(cut_f_signal)
    # y = cut_signal
    return y


def preprocessing_signal(data):
    res = data
    for channel in channels:
        signal = data[:, :, channel].ravel()
        filtered_signal = butter_bandpass_filter(signal, lowcut_freq, highcut_freq, freq)
        res[:, :, channel] = filtered_signal.reshape(data.shape[0], data.shape[1])
    return res


def preprocessing_signal_train_test(train, test):
    train_res = train
    test_res = test
    for channel in channels:
        # ButterWirth Filter
        train_signal = train[:, :, channel].ravel()
        filtered_train_signal = butter_bandpass_filter(train_signal, lowcut_freq, highcut_freq, freq)
        test_signal = test[:, :, channel].ravel()
        filtered_test_signal = butter_bandpass_filter(test_signal, lowcut_freq, highcut_freq, freq)

        # Normalizing
        scaler = StandardScaler()
        mean = filtered_train_signal.mean()
        std = filtered_train_signal.std()
        normalized_train_signal = (filtered_train_signal - mean) / std
        normalized_test_signal = (filtered_test_signal - mean) / std

        # scaler.fit(filtered_train_signal)
        #
        # filtered_train_signal = scaler.transform(filtered_train_signal)
        # filtered_test_signal = scaler.transform(filtered_test_signal)

        train_res[:, :, channel] = normalized_train_signal.reshape(train.shape[0], train.shape[1])
        test_res[:, :, channel] = normalized_test_signal.reshape(test.shape[0], test.shape[1])
    return train_res, test_res


# def Fit_scaler(data):
#     scaler.fit(data.reshape(data.shape[0], 1))
#
#
# def scaling_data(data, fit=False):
#     temp = data.ravel()
#     temp = temp.reshape(temp.shape[0], 1)
#     if fit:
#         temp = scaler.fit_transform(temp).ravel().reshape(data.shape[0], data.shape[1], data.shape[2])
#     else:
#         temp = scaler.transform(temp).ravel().reshape(data.shape[0], data.shape[1], data.shape[2])
#     return temp
#
#
# def scaling_train_test(train, test):
#     tr = train
#     tst = test
#     for ch in channels:
#         tr[:, :, ch] = scaler.fit_transform(tr[:, :, ch].ravel().reshape(tr.shape[0] * tr.shape[1], 1)).reshape(
#             tr.shape[0], tr.shape[1])
#         tst[:, :, ch] = scaler.transform(tst[:, :, ch].ravel().reshape(tst.shape[0] * tst.shape[1], 1)).reshape(
#             tst.shape[0], tst.shape[1])
#     return tr, tst


print('Loading The Data')

train = scipy.io.loadmat(r'../Data/Subject_A_Train.mat')
test = scipy.io.loadmat(r'../Data/Subject_A_Test.mat')
test_results = list(open('../Data/true_labels_A.txt', 'r').read())
train_characters = np.array(list(train['TargetChar'][0]))
test_characters = np.array(test_results)
print("Data is loaded")

matrix = np.array([['A', 'B', 'C', 'D', 'E', 'F'], ['G', 'H', 'I', 'J', 'K', 'L'], ['M', 'N', 'O', 'P', 'Q', 'R'],
                   ['S', 'T', 'U', 'V', 'W', 'X'], ['Y', 'Z', '1', '2', '3', '4'], ['5', '6', '7', '8', '9', '_']])

print("Processing Data....")
# processed_train, processed_test = preprocessing_signal_train_test(train['Signal'], test['Signal'])
# train['Signal'] = processed_train
# test['Signal'] = processed_test

# train['Signal'] = preprocessing_signal(train['Signal'])
#
# test['Signal'] = preprocessing_signal(test['Signal'])
#
# train['Signal'] = train['Signal'].astype('float16')
# test['Signal'] = test['Signal'].astype('float16')

print("Data Processed Successfully ")


# arr = np.array([])
# for c in [10]:
#     for epoch in range(train['Signal'].shape[0]):
#         for i in range(1,train['Signal'].shape[1]):
#             if(train['Flashing'][epoch][i] < 0.5 and train['Flashing'][epoch][i-1] > 0.5):
#                 sample = np.array(train['Signal'][epoch][i-24:i+window-24,channel])

#                 filteredSample = preprocessing_signal(sample)
#                 arr = np.append(arr,filteredSample)
# pprint(arr)            
# arr = scaler.fit_transform(arr)

#
def char_accuracy(X, characters, clf, prob_func=None):
    if prob_func is None:
        prob_func = clf.decision_function
    correct_predictions_count = 0
    for i in range(len(characters)):
        segment = X[i * 12:i * 12 + 12]
        score = prob_func(segment)
        row = np.argmax(score[6:]) + 7
        col = np.argmax(score[:6]) + 1
        predicted_char = matrix[row - 7][col - 1]
        target_char = characters[i]
        # print((i, target_char, predicted_char, row, col))
        if target_char == predicted_char:
            correct_predictions_count += 1
    return correct_predictions_count / len(characters)


def indeces_of_char(c):
    indeces = np.where(matrix == c)
    row = indeces[0][0] + 7
    col = indeces[1][0] + 1
    return row, col


def prepare_X_y_and_characters(data, characters):
    p300_res = np.zeros(window)
    nonp300_res = np.zeros(window)
    data_shape = data['Signal'].shape
    target = np.zeros(data_shape[0] * 12, int)
    responses = np.zeros((data_shape[0] * 12, window), float)
    for epoch in range(data_shape[0]):
        epoch_char = characters[epoch]
        row, col = indeces_of_char(epoch_char)
        target[12 * epoch + row - 1] = 1
        target[12 * epoch + col - 1] = 1

        for i in range(1, data_shape[1]):
            if data['Flashing'][epoch][i] < 0.5 < data['Flashing'][epoch][i - 1]:
                rowcol = int(data['StimulusCode'][epoch][i - 1])

                extracted_sample = data['Signal'][epoch][i - 24:i + window - 24, 10]
                responses[epoch * 12 + rowcol - 1] += extracted_sample
                if row == rowcol or col == rowcol:
                    p300_res += extracted_sample
                else:
                    nonp300_res += extracted_sample

    responses = responses / 15
    p300_res = p300_res / (2 * 15 * data_shape[0])
    nonp300_res = nonp300_res / (10 * 15 * data_shape[0])

    return responses, target, p300_res, nonp300_res


X_train, y_train, p300, nonP300 = prepare_X_y_and_characters(train, train_characters)
X_test, y_test, _, _ = prepare_X_y_and_characters(test, test_characters)
print(X_train[:100])


def prepare_balanced(X, y):
    p300_indices = np.where(y == 1)[0]
    non_p300_indices = np.where(y == 0)[0]
    np.random.seed(42)
    non_p300_indices = np.random.choice(non_p300_indices, size=p300_indices.shape[0] + 50, replace=False)
    pprint(p300_indices.shape)
    pprint(non_p300_indices.shape)

    X_balanced = np.append(X[p300_indices], X[non_p300_indices], axis=0)

    y_balanced = np.append(np.ones(p300_indices.shape[0]), np.zeros(non_p300_indices.shape[0]))

    pprint(X_balanced.shape)
    pprint(y_balanced.shape)

    return X_balanced, y_balanced


X_balanced, y_balanced = prepare_balanced(X_train, y_train)

def graphDrawer(arr, arr2):
    x_axis = np.array(range(window)) / 240
    plt.plot(x_axis, arr, color='#188038', label='P300')
    plt.plot(x_axis, arr2, color='#A1282C', label='Non-P300')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

graphDrawer(p300,nonP300)


print("Training")
svc_unbalanced = svm.SVC(kernel='rbf', probability=True)
svc_unbalanced.fit(X_train, y_train)
print("Score on training data SVM RBF unbalanced: {}".format(svc_unbalanced.score(X_train, y_train)))
print("Score on test data SVM RBF unbalanced: {}".format(svc_unbalanced.score(X_test, y_test)))
print("Done")
# Balabizo = lambda x: svc_unbalanced.predict_proba(x)[:, 1]
print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_unbalanced), "%")
# print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_unbalanced, Balabizo), "%")

print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_unbalanced), "%")

print("Training")
svc_balanced = svm.SVC(kernel='rbf', probability=True)
svc_balanced.fit(X_balanced, y_balanced)
print("Score on training data SVM RBF balanced: {}".format(svc_balanced.score(X_train, y_train)))
print("Score on test data SVM RBF balanced: {}".format(svc_balanced.score(X_test, y_test)))
print("Done")
#
print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_balanced), "%")
print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_balanced), "%")

print("Training")
svc_unbalanced = svm.SVC(kernel='linear', probability=True)
svc_unbalanced.fit(X_train, y_train)
print("Score on training data SVM linear unbalanced: {}".format(svc_unbalanced.score(X_train, y_train)))
print("Score on test data SVM linear unbalanced: {}".format(svc_unbalanced.score(X_test, y_test)))
print("Done")
# Balabizo = lambda x: svc_unbalanced.predict_proba(x)[:, 1]
print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_unbalanced), "%")
# print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_unbalanced, Balabizo), "%")

print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_unbalanced), "%")

print("Training")
svc_balanced = svm.SVC(kernel='linear', probability=True)
svc_balanced.fit(X_balanced, y_balanced)
print("Score on training data SVM linear balanced: {}".format(svc_balanced.score(X_train, y_train)))
print("Score on test data SVM linear balanced: {}".format(svc_balanced.score(X_test, y_test)))
print("Done")
#
print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_balanced), "%")
print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_balanced), "%")

print("Training")
svc_unbalanced = svm.SVC(kernel='sigmoid', probability=True)
svc_unbalanced.fit(X_train, y_train)
print("Score on training data SVM sigmoid unbalanced: {}".format(svc_unbalanced.score(X_train, y_train)))
print("Score on test data SVM sigmoid unbalanced: {}".format(svc_unbalanced.score(X_test, y_test)))
print("Done")
# Balabizo = lambda x: svc_unbalanced.predict_proba(x)[:, 1]
print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_unbalanced), "%")
# print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_unbalanced, Balabizo), "%")

print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_unbalanced), "%")

print("Training")
svc_balanced = svm.SVC(kernel='sigmoid', probability=True)
svc_balanced.fit(X_balanced, y_balanced)
print("Score on training data SVM sigmoid balanced: {}".format(svc_balanced.score(X_train, y_train)))
print("Score on test data SVM sigmoid balanced: {}".format(svc_balanced.score(X_test, y_test)))
print("Done")
#
print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_balanced), "%")
print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_balanced), "%")

print("Training LDA Unblalanced")
lda_unbalanced = LinearDiscriminantAnalysis()
lda_unbalanced.fit(X_train, y_train)

lda_score_train = lda_unbalanced.score(X_train, y_train)  # Coefficient of determination
lda_score_test = lda_unbalanced.score(X_test, y_test)

print("Score on training data LDA unbalanced: {}".format(lda_score_train))
print("Score on test data LDA unbalanced: {}".format(lda_score_test))
print("Done")

print("Train Character accuracy", char_accuracy(X_train, train_characters, lda_unbalanced), "%")
print("Test Character accuracy", char_accuracy(X_test, test_characters, lda_unbalanced), "%")

print("Training LDA blalanced")
lda_balanced = LinearDiscriminantAnalysis()
lda_balanced.fit(X_balanced, y_balanced)

lda_score_train = lda_balanced.score(X_train, y_train)  # Coefficient of determination
lda_score_test = lda_balanced.score(X_test, y_test)

print("Score on training data LDA balanced: {}".format(lda_score_train))
print("Score on test data LDA balanced: {}".format(lda_score_test))
print("Done")

print("Train Character accuracy", char_accuracy(X_train, train_characters, lda_balanced), "%")
print("Test Character accuracy", char_accuracy(X_test, test_characters, lda_balanced), "%")

print("Training Random Forest Unbalanced")

rf_unbalanced = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                                       max_features='sqrt', max_depth=90)

rf_unbalanced.fit(X_train, y_train);
rf_score_train = rf_unbalanced.score(X_train, y_train)
rf_score_test = rf_unbalanced.score(X_test, y_test)

print('Score on training data Random Forest unbalanced:: {}'.format(rf_score_train))
print('test score: {}'.format(rf_score_test))

Balabizo = lambda x: rf_unbalanced.predict_proba(x)[:, 1]

print("Train Character accuracy", char_accuracy(X_train, train_characters, rf_unbalanced, Balabizo), "%")
print("Test Character accuracy", char_accuracy(X_test, test_characters, rf_unbalanced, Balabizo), "%")

print("Training Random Forest balanced")

rf_balanced = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                                     max_features='sqrt', max_depth=90)

rf_balanced.fit(X_train, y_train);
rf_score_train = rf_balanced.score(X_balanced, y_balanced)
rf_score_test = rf_balanced.score(X_test, y_test)

print('Score on training data Random Forest balanced:: {}'.format(rf_score_train))
print('test score: {}'.format(rf_score_test))

Balabizo = lambda x: rf_balanced.predict_proba(x)[:, 1]

print("Train Character accuracy", char_accuracy(X_train, train_characters, rf_balanced, Balabizo), "%")
print("Test Character accuracy", char_accuracy(X_test, test_characters, rf_balanced, Balabizo), "%")
