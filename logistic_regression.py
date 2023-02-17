import numpy as np
import pandas as pd
import os
from tkinter.filedialog import askopenfilename


def load_features():
    # try:
    #     filename = askopenfilename(initialdir=os.getcwd(), title="Select file")
    #     if filename is None:
    #         return
    # except AttributeError:
    #     return

    filename = r'C:\Users\Matt\PycharmProjects\calibur\log_test.csv'

    try:
        file1 = pd.read_csv(filename)
    except IndexError:
        print('Please load an appropriately formatted feature array file')
        return

    features = file1.values
    bin_labels = []
    bin_reps = []
    feature_array = []
    for i in range(0, np.size(features, 0)):
        bin_reps.append(int(features[i, -1]))
        bin_labels.append(int(features[i, -2]))
        feature_array.append(features[i, :-2])

    return np.array(feature_array), np.array(bin_labels), np.array(bin_reps)


def split_folds(features, labels, bin_reps):
    folds = []
    for j in np.unique(np.array(reps)):
        features_test = []
        features_train = []
        labels_test = []
        labels_train = []
        for i in range(0, np.size(features, 0)):
            if bin_reps[i] == j:
                features_test.append(features[i, :])
                labels_test.append(labels[i])
            else:
                features_train.append(features[i, :])
                labels_train.append(labels[i])

        folds.append((np.array(features_train), np.array(labels_train).reshape((len(labels_train), 1)), np.array(features_test), np.array(labels_test).reshape((len(labels_test), 1))))

    return folds


def train(X, Y, alpha, L, iterations):
    n = X.shape[0]
    w = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1,1)

    for iteration in range(0, iterations):
        Z = np.dot(X, w) + b
        # Z = np.array(Z, dtype='float64')
        A = 1/(1+np.exp(-Z))
        dw = (1/n)*(np.dot(X.T, (A - Y)) + L*w)
        db = (1/n)*np.sum(A - Y)
        w = w - alpha*dw
        b = b - alpha*db

    return w, b


def predict(X, w, b):
    Z = np.dot(X, w) + b
    # Z = np.array(Z, dtype='float64')
    return 1/(1+np.exp(-Z))


def evaluate(Y_hat, Y, threshold):
    y2 = []
    for y in Y_hat:
        if y > threshold:
            y2.append(1)
        elif y <= 1 - threshold:
            y2.append(0)
        else:
            y2.append(0)

    # y2 = np.round(Y_hat)
    y2 = np.array(y2).reshape((len(y2), 1))
    i = np.where(y2 == Y)[0]

    return np.size(i)/np.size(Y)


if __name__ == '__main__':
    [X, Y, reps] = load_features()
    folds = split_folds(X, Y, reps)
    cv_accuracies = []
    for i, fold in enumerate(folds):
        Xtrain = fold[0]
        scale_factors = np.max(Xtrain, axis=0)
        Xtrain = Xtrain/scale_factors
        Ytrain = fold[1]
        Xtest = fold[2]/scale_factors
        Ytest = fold[3]
        [w, b] = train(Xtrain, Ytrain, alpha=0.001, L=0, iterations=25000)
        Y_hat = predict(Xtest, w, b)
        accuracy = evaluate(Y_hat, Ytest, 0.5)
        print("Fold {}: ".format(str(i)) + str(accuracy))
        cv_accuracies.append(accuracy)

    print("Mean Accuracy: " + str(np.mean(cv_accuracies)))
    print("STD Accuracy: " + str(np.std(cv_accuracies)))