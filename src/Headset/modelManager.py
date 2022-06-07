import scipy.io
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

from ConfigUnicorn import *
from PreprocessingUnicorn import preprocessing, preprocessing_Raw
import numpy as np


def instruction_accuracy(X, characters, clf, prob_func=None):
    if prob_func is None:
        prob_func = clf.decision_function
    correct_predictions_count = 0
    for i in range(len(characters)):
        segment = X[i * NUMBER_OF_FLASHES:i * NUMBER_OF_FLASHES + NUMBER_OF_FLASHES]
        score = prob_func(segment)
        row = np.argmax(score[3:]) + 4
        col = np.argmax(score[:3]) + 1
        predicted_char = INSTRUCTION_MATRIX[row - 7][col - 1]
        target_char = characters[i]
        # print((i, target_char, predicted_char, row, col))
        if target_char == predicted_char:
            correct_predictions_count += 1
    return correct_predictions_count / len(characters)


def indeces_of_instruction(c):
    indeces = np.where(INSTRUCTION_MATRIX == c)
    row = indeces[0][0] + 4
    col = indeces[1][0] + 1
    return row, col


def prepare_for_model(signal, code, targeted):
    p300_res = np.zeros(WINDOW * NUMBER_OF_CHANNELS)
    nonp300_res = np.zeros(WINDOW * CHANNELS.shape[0])
    target = np.zeros(NUMBER_OF_EPOCHS * NUMBER_OF_FLASHES, int)
    responses = np.zeros((NUMBER_OF_EPOCHS * NUMBER_OF_FLASHES, WINDOW * NUMBER_OF_CHANNELS), np.float32)
    epoch = 0
    number_of_trial = int(0)
    instructions = []
    current_instruction = ""
    targeted_row, targeted_col = 0, 0
    for i in range(1, signal.shape[0]):
        if code[i - 1] < 0.5 < code[i]:
            if number_of_trial == 0:
                current_instruction = targeted[i]
                instructions.append(current_instruction)
                targeted_row, targeted_col = indeces_of_instruction(current_instruction)
                target[NUMBER_OF_FLASHES * epoch + targeted_row - 1] = 1
                target[NUMBER_OF_FLASHES * epoch + targeted_col - 1] = 1
            number_of_trial += 1
            rowcol = int(code[i])

            for ch in CHANNELS:
                extracted_sample = signal[i:i + WINDOW, ch]
                responses[epoch * NUMBER_OF_FLASHES + rowcol - 1][
                ch * WINDOW:ch * WINDOW + WINDOW] += extracted_sample
                if targeted_col == rowcol or targeted_row == rowcol:
                    p300_res[ch * WINDOW:ch * WINDOW + WINDOW] += extracted_sample
                else:
                    nonp300_res[ch * WINDOW:ch * WINDOW + WINDOW] += extracted_sample
            if number_of_trial == NUMBER_OF_FLASHES * NUMBER_OF_TRIALS:
                number_of_trial = 0
                epoch += 1
    responses = responses / NUMBER_OF_EPOCHS
    p300_res = p300_res / (2 * NUMBER_OF_TRIALS * NUMBER_OF_EPOCHS)
    nonp300_res = nonp300_res / (4 * NUMBER_OF_TRIALS * NUMBER_OF_EPOCHS)
    return responses, target, np.array(instructions), p300_res, nonp300_res


def trainModels(X, y, instructions):
    print("Training")
    svc_unbalanced = svm.SVC(C=10, gamma=0.001, kernel='rbf', probability=True)
    svc_unbalanced.fit(X, y)
    print("Score on training data SVM RBF unbalanced: {}".format(svc_unbalanced.score(X, y)))
    print("Done")
    print("Train Character accuracy", instruction_accuracy(X, instructions, svc_unbalanced), "%")
    dump(svc_unbalanced, f"./experiment/{NAME}/Models/SVC_RBF.pkl")
    # Balabizo = lambda x: svc_unbalanced.predict_proba(x)[:, 1]
    # print("Train Character accuracy", char_accuracy(X, train_characters, svc_unbalanced, Balabizo), "%")

    # print("Training")
    # svc_balanced = svm.SVC(C=10, gamma=0.001, kernel='rbf', probability=True)
    # svc_balanced.fit(X_balanced, y_balanced)
    # print("Score on training data SVM RBF balanced: {}".format(svc_balanced.score(X, y)))
    # print("Score on test data SVM RBF balanced: {}".format(svc_balanced.score(X_test, y_test)))
    # print("Done")
    # print("Train Character accuracy", char_accuracy(X, train_characters, svc_balanced), "%")
    # # print("Train Character accuracy", char_accuracy(X, train_characters, svc_unbalanced, Balabizo), "%")
    # print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_balanced), "%")

    print("_________________________")

    print("Training")
    svc_unbalanced = svm.SVC(C=0.1, gamma=1, kernel='linear', probability=True)
    svc_unbalanced.fit(X, y)
    print("Score on training data SVM linear unbalanced: {}".format(svc_unbalanced.score(X, y)))
    print("Done")
    # Balabizo = lambda x: svc_unbalanced.predict_proba(x)[:, 1]
    print("Train Character accuracy", instruction_accuracy(X, instructions, svc_unbalanced), "%")
    # print("Train Character accuracy", char_accuracy(X, train_characters, svc_unbalanced, Balabizo), "%")
    dump(svc_unbalanced, f"./experiment/{NAME}/Models/SVC_Linear.pkl")
    print("_________________________")

    # print("Training")
    # svc_balanced = svm.SVC(C=0.1, gamma=1, kernel='linear', probability=True)
    # svc_balanced.fit(X_balanced, y_balanced)
    # print("Score on training data SVM linear balanced: {}".format(svc_balanced.score(X, y)))
    # print("Score on test data SVM linear balanced: {}".format(svc_balanced.score(X_test, y_test)))
    # print("Done")
    # #
    # print("Train Character accuracy", char_accuracy(X, train_characters, svc_balanced), "%")
    # print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_balanced), "%")
    # from joblib import dump
    # dump(svc_balanced,"model.joblib")
    #
    # print("_________________________")

    print("Training")
    svc_unbalanced = svm.SVC(kernel='sigmoid', probability=True)
    svc_unbalanced.fit(X, y)
    print("Score on training data SVM sigmoid unbalanced: {}".format(svc_unbalanced.score(X, y)))
    print("Done")
    # Balabizo = lambda x: svc_unbalanced.predict_proba(x)[:, 1]
    print("Train Character accuracy", instruction_accuracy(X, instructions, svc_unbalanced), "%")
    # print("Train Character accuracy", char_accuracy(X, train_characters, svc_unbalanced, Balabizo), "%")
    dump(svc_unbalanced, f"./experiment/{NAME}/Models/SVC_Sigmoid.pkl")
    print("_________________________")

    # print("Training")
    # svc_balanced = svm.SVC(kernel='sigmoid', probability=True)
    # svc_balanced.fit(X_balanced, y_balanced)
    # print("Score on training data SVM sigmoid balanced: {}".format(svc_balanced.score(X, y)))
    # print("Score on test data SVM sigmoid balanced: {}".format(svc_balanced.score(X_test, y_test)))
    # print("Done")
    # #
    # print("Train Character accuracy", char_accuracy(X, train_characters, svc_balanced), "%")
    # print("Test Character accuracy", char_accuracy(X_test, test_characters, svc_balanced), "%")
    #
    # print("_________________________")

    print("Training LDA Unblalanced")
    lda_unbalanced = LinearDiscriminantAnalysis()
    lda_unbalanced.fit(X, y)

    lda_score_train = lda_unbalanced.score(X, y)  # Coefficient of determination

    print("Score on training data LDA unbalanced: {}".format(lda_score_train))
    print("Done")

    print("Train Character accuracy", instruction_accuracy(X, instructions, lda_unbalanced), "%")
    dump(lda_unbalanced, f"./experiment/{NAME}/Models/LDA.pkl")
    print("_________________________")

    # print("Training LDA blalanced")
    # lda_balanced = LinearDiscriminantAnalysis()
    # lda_balanced.fit(X_balanced, y_balanced)
    #
    # lda_score_train = lda_balanced.score(X, y)  # Coefficient of determination
    # lda_score_test = lda_balanced.score(X_test, y_test)
    #
    # print("Score on training data LDA balanced: {}".format(lda_score_train))
    # print("Score on test data LDA balanced: {}".format(lda_score_test))
    # print("Done")
    #
    # print("Train Character accuracy", char_accuracy(X, train_characters, lda_balanced), "%")
    # print("Test Character accuracy", char_accuracy(X_test, test_characters, lda_balanced), "%")
    #
    # print("_________________________")

    print("Training Random Forest Unbalanced")

    rf_unbalanced = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=5, min_samples_leaf=2,
                                           max_features='sqrt', max_depth=90)

    rf_unbalanced.fit(X, y);
    rf_score_train = rf_unbalanced.score(X, y)

    print('Score on training data Random Forest unbalanced:: {}'.format(rf_score_train))

    Balabizo = lambda x: rf_unbalanced.predict_proba(x)[:, 1]

    print("Train Character accuracy", instruction_accuracy(X, instructions, rf_unbalanced, Balabizo), "%")
    dump(rf_unbalanced, f"./experiment/{NAME}/Models/Random_Forest.pkl")
    print("_________________________")

    # print("Training Random Forest balanced")
    #
    # rf_balanced = RandomForestClassifier(random_state=42, n_estimators=200, min_samples_split=5, min_samples_leaf=2,
    #                                      max_features='sqrt', max_depth=90)
    #
    # rf_balanced.fit(X_balanced, y_balanced)
    # rf_score_train = rf_balanced.score(X_balanced, y_balanced)
    # rf_score_test = rf_balanced.score(X_test, y_test)
    #
    # print('Score on training data Random Forest balanced:: {}'.format(rf_score_train))
    # print('test score: {}'.format(rf_score_test))
    #
    # Balabizo = lambda x: rf_balanced.predict_proba(x)[:, 1]
    #
    # print("Train Character accuracy", char_accuracy(X, train_characters, rf_balanced, Balabizo), "%")
    # print("Test Character accuracy", char_accuracy(X_test, test_characters, rf_balanced, Balabizo), "%")


def testModels(X, y,instructions):
    svc_unbalanced = load(f"./experiment/{NAME}/Models/SVC_RBF.pkl")
    print("Testing SVC RBF")
    print("Score on test data SVM RBF unbalanced: {}".format(svc_unbalanced.score(X, y)))
    print("Test instruction accuracy", instruction_accuracy(X, instructions, svc_unbalanced), "%")

    print("_______________________")

    svc_unbalanced = load(f"./experiment/{NAME}/Models/SVC_Linear.pkl")
    print("Testing SVC Linear")
    print("Score on test data SVM Linear unbalanced: {}".format(svc_unbalanced.score(X, y)))
    print("Test instruction accuracy", instruction_accuracy(X, instructions, svc_unbalanced), "%")

    print("_______________________")

    svc_unbalanced = load(f"./experiment/{NAME}/Models/SVC_Sigmoid.pkl")
    print("Testing SVC Sigmoid")
    print("Score on test data SVM Sigmoid unbalanced: {}".format(svc_unbalanced.score(X, y)))
    print("Test instruction accuracy", instruction_accuracy(X, instructions, svc_unbalanced), "%")

    print("_______________________")

    lda_unbalanced = load(f"./experiment/{NAME}/Models/LDA.pkl")
    print("Testing LDA")
    print("Score on test data LDA unbalanced: {}".format(lda_unbalanced.score(X, y)))
    print("Test instruction accuracy", instruction_accuracy(X, instructions, lda_unbalanced), "%")

    print("_______________________")

    rf_unbalanced = load(f"./experiment/{NAME}/Models/Random_Forest.pkl")

    rf_score_train = rf_unbalanced.score(X, y)

    print('Score on training data Random Forest unbalanced:: {}'.format(rf_score_train))

    Balabizo = lambda x: rf_unbalanced.predict_proba(x)[:, 1]

    print("Twst instruction accuracy", instruction_accuracy(X, instructions, rf_unbalanced, Balabizo), "%")

    print("_________________________")



def main(train_or_test):
    print('Loading The Data')
    all_data = scipy.io.loadmat(rf'../experiment/{NAME}/{NAME}{train_or_test}.mat')
    # {'Samples': samples, 'col': rows, 'target': targets})
    data = np.array(all_data['Samples'])

    print("Processing Data....")
    preprocessing_Raw(data)
    preprocessing(data)
    X, y, instructions, p300, nonP300 = prepare_for_model(data, all_data['StimulusCode'], all_data['TargetInstruction'])

    from scipy.io import savemat
    from os import mkdir, path
    if not path.isdir(f"./experiment/{NAME}/Models"):
        mkdir(f"./experiment/{NAME}/Models")


    if train_or_test == "train":
        trainModels(X, y, instructions)
    else:
        testModels(X, y, instructions)


main("train")
