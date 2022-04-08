import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.io
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

from sklearn.preprocessing import normalize
from scipy.stats.mstats import winsorize

from pprint import pprint

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

from scipy.fft import rfft, irfft, rfftfreq, fft, fftfreq, ifft

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
        print((i, target_char, predicted_char, row, col))
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

# def graphDrawer(arr, arr2):
#     x_axis = np.array(range(window)) / 240
#     plt.plot(x_axis, arr, color='#188038', label='P300')
#     plt.plot(x_axis, arr2, color='#A1282C', label='Non-P300')
#
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
# graphDrawer(p300,nonP300)


print("Training")
svc_unbalanced = svm.SVC(kernel='rbf', probability=True)
svc_unbalanced.fit(X_train, y_train)
print("Score on training data SVM RBF unbalanced: {}".format(svc_unbalanced.score(X_train, y_train)))
print("Score on test data SVM RBF unbalanced: {}".format(svc_unbalanced.score(X_test, y_test)))
print("Done")
Balabizo = lambda x: svc_unbalanced.predict_proba(x)[:, 1]
print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_unbalanced), "%")
print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_unbalanced, Balabizo), "%")

# print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_unbalanced), "%")

print("Training")
svc_balanced = svm.SVC(kernel='rbf', probability=True)
svc_balanced.fit(X_balanced, y_balanced)
print("Score on training data SVM RBF balanced: {}".format(svc_balanced.score(X_train, y_train)))
print("Score on test data SVM RBF balanced: {}".format(svc_balanced.score(X_test, y_test)))
print("Done")
#
# print("Train Character accuracy", char_accuracy(X_train, train_characters, svc_balanced), "%")
# print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_balanced), "%")

# svc_score_train = clf.decision_function(X)
# svc_score_test =clf.predict(X)

# def characters_accuracy(clf, df, prob_func=None):
#     if (prob_func == None):
#         prob_func = clf.decision_function
#     res = np.array([])
#     for epoch in range(df.min()["Epoch"], df.max()["Epoch"] + 1):
#         df_epoch = df[df['Epoch'] == epoch].reset_index(drop=True)
#         if (df_epoch.size == 0):
#             continue
#         X, _ = prepare_X_y(df_epoch)
#         df_epoch = df_epoch[["Epoch", "Character", "rowcol", "Isp300"]]
#         df_epoch['Score'] = prob_func(X)
#         row = df_epoch.iloc[6:]['Score'].idxmax()
#         col = df_epoch.iloc[:6]['Score'].idxmax()
#         predicted_row = df_epoch.iloc[row]['rowcol']
#         predicted_col = df_epoch.iloc[col]['rowcol']
#         predicted_char = matrix[row - 6][col]
#         target_char = df_epoch['Character'][0]
#         res = np.append(res, {'Epoch': epoch, 'Target_Character': target_char, 'Predicted_Character': predicted_char,
#                               'Predicted_Row': predicted_row, 'Predicted_Col': predicted_col,
#                               'Same': (target_char == predicted_char)})
#         df_res = pd.DataFrame(list(res), columns=['Epoch', 'Target_Character', 'Predicted_Character', 'Predicted_Row',
#                                                   'Predicted_Col', 'Same'])
#     return metrics.accuracy_score(df_res['Target_Character'], df_res['Predicted_Character']), df_res


# responses = np.array([])
# # samples = np.empty((0,window*channels.shape[0]), int)
# for epoch in range(train['Signal'].shape[0]):
#     for i in range(1, train['Signal'].shape[1]):
#         if (train['Flashing'][epoch][i] < 0.5 and train['Flashing'][epoch][i - 1] > 0.5):
#             sample = np.array([])
#
#             for ch in channels:
#                 sample = np.append(sample, train['Signal'][epoch][i - 24:i + window - 24, 10])
#             #             samples = np.append(samples,[sample],axis = 0)
#             rowcol = train['StimulusCode'][epoch][i - 1]
#             temp = {'Epoch': epoch + 1, 'Character': Characters[epoch], 'rowcol': rowcol, 'Sample': sample,
#                     'Isp300': train['StimulusType'][epoch][i - 1]}
#             #             for n in range(sample.shape[0]):
#             #                 temp[n] = sample[n]
#             responses = np.append(responses, temp)
#
# # df = pd.DataFrame(list(responses),columns =['Epoch','Character', 'rowcol','Isp300']+list(range(window*channels.shape[0])))
# df = pd.DataFrame(list(responses), columns=['Epoch', 'Character', 'rowcol', 'Sample', 'Isp300'])
#
# df.head()
#
#
# # df_samples = pd.DataFrame(list(samples),columns = list(range(window*channels.shape[0])))
# # df_samples.head()
#
#


# def character_in_Row_Col(c, n):
#     if n not in (range(1, 13)):
#         return 0.0
#     if n <= 6:
#         if (c in matrix[:, n - 1]):
#             return 1.0
#         else:
#             return 0.0
#     else:
#         if (c in matrix[n - 7]):
#             return 1.0
#         else:
#             return 0.0
# #
#
# test_responses = np.array([])
# for epoch in range(len(test_results)):
#     for i in range(1, test['Signal'].shape[1]):
#         if (test['Flashing'][epoch][i] < 0.5 and test['Flashing'][epoch][i - 1] > 0.5):
#             sample = np.array([])
#             for ch in channels:
#                 sample = np.append(sample, test['Signal'][epoch][i - 24:i + window - 24, ch])
#             rowcol = test['StimulusCode'][epoch][i - 1]
#             temp = {'Epoch': epoch + 1, 'Character': test_results[epoch], 'rowcol': rowcol, 'Sample': sample,
#                     'Isp300': character_in_Row_Col(test_results[epoch], int(rowcol))}
#             #             for n in range(sample.shape[0]):
#             #                 temp[n] = sample[n]
#             test_responses = np.append(test_responses, temp)
#
# # df_test = pd.DataFrame(list(test_responses),columns =['Epoch','Character', 'rowcol','Isp300']+list(range(window*channels.shape[0])))
# df_test = pd.DataFrame(list(test_responses), columns=['Epoch', 'Character', 'rowcol', 'Sample', 'Isp300'])
# df_test['Epoch'] = df_test['Epoch'] + 85
# df_test.head()
#
#
# # df_all = pd.concat([df,df_test]).reset_index()
#
#
# def graphDrawer(df):
#     average_p300 = df[df['Isp300'] == 1]['Sample'].mean()
#     average_non_p300 = df[df['Isp300'] == 0]['Sample'].mean()
#     x_axis = np.array(range(window * channels.shape[0]))
#     plt.plot(x_axis, average_p300, color='#188038', label='P300')
#     plt.plot(x_axis, average_non_p300, color='#A1282C', label='Non-P300')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.title('P300 VS Non-P300')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# graphDrawer(df)
#
# graphDrawer(df_test)
#
# # graphDrawer(df_all)
#
#
# df = pd.concat([df, pd.DataFrame(list(map(np.ravel, (list(df['Sample'])))))], axis=1)
#
# df_test = pd.concat([df_test, pd.DataFrame(list(map(np.ravel, (list(df_test['Sample'])))))], axis=1)
#
# df_test = df_test.groupby(['Epoch', 'Character', 'rowcol', 'Isp300']).mean().sort_values(
#     by=['Epoch', 'rowcol']).reset_index()
#
# df = df.groupby(['Epoch', 'Character', 'rowcol', 'Isp300']).mean().sort_values(by=['Epoch', 'rowcol']).reset_index()
#
# df_all = pd.concat([df, df_test]).reset_index()
#
# p300_df = df[df['Isp300'] == 1.0].reset_index()
# non_p300_df = df[df['Isp300'] == 0.0].reset_index()
#
#
# # In[83]:
#
#
# def characters_accuracy(clf, df, prob_func=None):
#     if (prob_func == None):
#         prob_func = clf.decision_function
#     res = np.array([])
#     for epoch in range(df.min()["Epoch"], df.max()["Epoch"] + 1):
#         df_epoch = df[df['Epoch'] == epoch].reset_index(drop=True)
#         if (df_epoch.size == 0):
#             continue
#         X, _ = prepare_X_y(df_epoch)
#         df_epoch = df_epoch[["Epoch", "Character", "rowcol", "Isp300"]]
#         df_epoch['Score'] = prob_func(X)
#         row = df_epoch.iloc[6:]['Score'].idxmax()
#         col = df_epoch.iloc[:6]['Score'].idxmax()
#         predicted_row = df_epoch.iloc[row]['rowcol']
#         predicted_col = df_epoch.iloc[col]['rowcol']
#         predicted_char = matrix[row - 6][col]
#         target_char = df_epoch['Character'][0]
#         res = np.append(res, {'Epoch': epoch, 'Target_Character': target_char, 'Predicted_Character': predicted_char,
#                               'Predicted_Row': predicted_row, 'Predicted_Col': predicted_col,
#                               'Same': (target_char == predicted_char)})
#         df_res = pd.DataFrame(list(res), columns=['Epoch', 'Target_Character', 'Predicted_Character', 'Predicted_Row',
#                                                   'Predicted_Col', 'Same'])
#     return metrics.accuracy_score(df_res['Target_Character'], df_res['Predicted_Character']), df_res
#
#
# # In[98]:
#
#
# from sklearn.decomposition import PCA
#
# pca = PCA(100)
#
#
# def prepare_X_y(temp_df, fit=False):
#     X = temp_df.drop(columns=['Epoch', 'Character', 'rowcol', 'Isp300'])
#     if fit:
#         scaler.fit(X)
#         pca.fit(X)
#
#         X = scaler.transform(X)
#         X = pca.transform(X)
#     y = temp_df['Isp300']
#     return X, y
#
#
# # In[ ]:
#
#
# df_balanced = pd.concat([df[df['Isp300'] == 1], df[df['Isp300'] == 0].sample(170, random_state=42, ignore_index=True)],
#                         ignore_index=True)
# X_train, y_train = prepare_X_y(df_balanced, True)
#
# # X_train,y_train = prepare_X_y(df,True)
# # X_train = pca.fit_transform(X_train)
#
#
# X_test, y_test = prepare_X_y(df_test)
# # X_test = pca.transform(X_test)
#
# # df_balanced = pd.concat([df[df['Isp300']==1],df[df['Isp300']==0].sample(170,random_state = 42,ignore_index = True)],ignore_index = True)
# # X_balanced,y_balanced = prepare_X_y(df_balanced)
# X_balanced, y_balanced = X_train, y_train
#
# print(X_train.shape)
# print(X_test.shape)
#
# # In[94]:
#
#
# svc_unbalanced = svm.SVC(kernel='linear', probability=True)
# ##clf = GradientBoostingClassifier(n_estimators=10000)
# svc_unbalanced.fit(X_train, y_train)
# print("Score on training data: {}".format(svc_unbalanced.score(X_train, y_train)))
# print("Score on test data: {}".format(svc_unbalanced.score(X_test, y_test)))
# # svc_score_train = clf.decision_function(X)
# # svc_score_test =clf.predict(X)
#
#
# # In[ ]:
#
#
# accuracy, _ = characters_accuracy(svc_unbalanced, df)
# accuracy
#
# # In[ ]:
#
#
# accuracy, _ = characters_accuracy(svc_unbalanced, df_test)
#
# accuracy
#
# # In[36]:
#
#

#
#
# # In[37]:
#
#
# accuracy, _ = characters_accuracy(svc_unbalanced, df)
# accuracy
#
# # In[38]:
#
#
# accuracy, _ = characters_accuracy(svc_unbalanced, df_test)
#
# accuracy
#
# # In[39]:
#
#
# svc_balanced = svm.SVC(kernel='sigmoid')
# svc_balanced.fit(X_balanced, y_balanced)
#
# svc_score_train = svc_balanced.score(X_train, y_train)  # Coefficient of determination
# svc_score_test = svc_balanced.score(X_test, y_test)
#
# # y_pred=svc.predict(X_test)
# print('train score: {}'.format(svc_score_train))
# print('test score: {}'.format(svc_score_test))
#
# # In[40]:
#
#
# a, _ = characters_accuracy(svc_balanced, df)
# a
#
# # In[41]:
#
#
# a, _ = characters_accuracy(svc_balanced, df_test)
# a
#
# # In[111]:
#
#
# svc_balanced = svm.SVC(kernel='linear')
# svc_balanced.fit(X_balanced, y_balanced)
#
# svc_score_train = svc_balanced.score(X_train, y_train)  # Coefficient of determination
# svc_score_test = svc_balanced.score(X_test, y_test)
#
# # y_pred=svc.predict(X_test)
# print('train score: {}'.format(svc_score_train))
# print('test score: {}'.format(svc_score_test))
#
# # In[112]:
#
#
# a, _ = characters_accuracy(svc_balanced, df)
# a
#
# # In[113]:
#
#
# a, _ = characters_accuracy(svc_balanced, df_test)
# a
#
# # In[114]:
#
#
# svc_balanced = svm.SVC(kernel='rbf')
# svc_balanced.fit(X_balanced, y_balanced)
#
# svc_score_train = svc_balanced.score(X_train, y_train)  # Coefficient of determination
# svc_score_test = svc_balanced.score(X_test, y_test)
#
# # y_pred=svc.predict(X_test)
# print('train score: {}'.format(svc_score_train))
# print('test score: {}'.format(svc_score_test))
#
# # In[115]:
#
#
# a, _ = characters_accuracy(svc_balanced, df)
# a
#
# # In[116]:
#
#
# a, _ = characters_accuracy(svc_balanced, df_test)
# a
#
# # In[117]:
#
#
# svc_balanced = svm.SVC(kernel='sigmoid')
# svc_balanced.fit(X_balanced, y_balanced)
#
# svc_score_train = svc_balanced.score(X_train, y_train)  # Coefficient of determination
# svc_score_test = svc_balanced.score(X_test, y_test)
#
# # y_pred=svc.predict(X_test)
# print('train score: {}'.format(svc_score_train))
# print('test score: {}'.format(svc_score_test))
#
# # In[118]:
#
#
# a, _ = characters_accuracy(svc_balanced, df)
# a
#
# # In[119]:
#
#
# a, _ = characters_accuracy(svc_balanced, df_test)
# a
#
# # ## LDA  Linear Discriminant Analysis
#
# # In[120]:
#
#
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# # In[121]:
#
#
# lda_unbalanced = LinearDiscriminantAnalysis()
# lda_unbalanced.fit(X_train, y_train)
#
# lda_score_train = lda_unbalanced.score(X_train, y_train)  # Coefficient of determination
# lda_score_test = lda_unbalanced.score(X_test, y_test)
#
# # y_pred=svc.predict(X_test)
# print('train score: {}'.format(svc_score_train))
# print('test score: {}'.format(svc_score_test))
#
# # In[122]:
#
#
# a, _ = characters_accuracy(lda_unbalanced, df)
# a
#
# # In[123]:
#
#
# a, _ = characters_accuracy(lda_unbalanced, df_test)
# a
#
# # ## Random Forest
#
# # In[124]:
#
#
# from sklearn.ensemble import RandomForestClassifier
#
# # In[125]:
#
#
# rf = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=5, min_samples_leaf=2,
#                             max_features='sqrt', max_depth=90)
#
# rf.fit(X_train, y_train);
# rf_score_train = rf.score(X_train, y_train)  # Coefficient of determination
# rf_score_test = rf.score(X_test, y_test)
#
# # y_pred=svc.predict(X_test)
# print('train score: {}'.format(rf_score_train))
# print('test score: {}'.format(rf_score_test))
#
# # In[126]:
#
#
# Balabizo = lambda x: rf.predict_proba(x)[:, 1]
# a, _ = characters_accuracy(lda_unbalanced, df_test, Balabizo)
# a
#
# # In[127]:
#
#
# rf_balanced = RandomForestClassifier(random_state=42, n_estimators=275, min_samples_split=5, min_samples_leaf=4,
#                                      max_features='sqrt', max_depth=40)
#
# rf.fit(X_balanced, y_balanced);
# rf_score_train = rf.score(X_train, y_train)  # Coefficient of determination
# rf_score_test = rf.score(X_test, y_test)
#
# # y_pred=svc.predict(X_test)
# print('train score: {}'.format(rf_score_train))
# print('test score: {}'.format(rf_score_test))
#
# # In[128]:
#
#
# Balabizo = lambda x: rf.predict_proba(x)[:, 1]
# a, _ = characters_accuracy(lda_unbalanced, df_test, Balabizo)
# a
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# from sklearn.model_selection import RandomizedSearchCV
#
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=200, stop=500, num=5)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 7]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
#
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
#
# pprint(random_grid)
#
# # In[286]:
#
#
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier(random_state=42)
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
#                                n_iter=100, scoring='neg_mean_absolute_error',
#                                cv=3, verbose=2, random_state=42, n_jobs=-1,
#                                return_train_score=True)
#
# # Fit the random search model
# rf_random.fit(X_balanced, y_balanced)
