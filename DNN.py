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

def train(X, Y, alpha, n_hidden_layers, n_nodes, iterations):
    if len(n_nodes) < n_hidden_layers:
        print("Please specify number of nodes in each layer.")
        return
    X = X.T
    Y = Y.T
    m = X.shape[1]
    n_temp = [X.shape[0]]
    n_temp.extend(n_nodes)
    n_temp.extend([1])
    n_nodes = n_temp

    W = [0]
    b = [0]
    for i in range(1, n_hidden_layers + 2):
        W.append(np.random.randn(n_nodes[i], n_nodes[i-1]) * 0.01)
        b.append(np.zeros((n_nodes[i], 1)))

    for iteration in range(0, iterations):
        A = [X]
        for i in range(1, n_hidden_layers + 1):
            Z = np.dot(W[i], A[i-1]) + b[i]
            A.append((np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z)))
        Z = np.dot(W[n_hidden_layers + 1], A[n_hidden_layers]) + b[n_hidden_layers + 1]
        A.append(1 / (1 + np.exp(-Z)))

        dZ = A[n_hidden_layers + 1] - Y
        dW = [(1/m)*np.dot(dZ, A[n_hidden_layers].T)]
        db = [(1/m)*np.sum(dZ, axis=1, keepdims=True)]
        for i in range(n_hidden_layers, 0, -1):
            dZ = np.dot(W[i+1].T, dZ) * (1 - np.power(A[i], 2))
            dW.append((1/m)*np.dot(dZ, A[i-1].T))
            db.append((1/m)*np.sum(dZ, axis=1, keepdims=True))

        dW.append(0)
        db.append(0)
        dW.reverse()
        db.reverse()

        for i in range(1, len(W)):
            W[i] = W[i] - alpha*dW[i]
            b[i] = b[i] - alpha*db[i]

    return W, b

def predict(X, W, b):
    X = X.T
    A = [X]

    n_hidden_layers = np.size(b) - 2
    for i in range(1, n_hidden_layers+1):
        Z = np.dot(W[i], A[i-1]) + b[i]
        A.append((np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z)))
    Z = np.dot(W[n_hidden_layers + 1], A[n_hidden_layers]) + b[n_hidden_layers + 1]
    A.append(1 / (1 + np.exp(-Z)))
    return A[n_hidden_layers + 1].reshape((1, X.shape[1]))

def evaluate(Y_hat, Y, threshold):
    y2 = []
    for y in Y_hat[0]:
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
        [W, b] = train(Xtrain, Ytrain, alpha=0.001, n_hidden_layers=1, n_nodes=[4], iterations=25000)
        Y_hat = predict(Xtest, W, b)
        accuracy = evaluate(Y_hat, Ytest, 0.5)
        print("Fold {}: ".format(str(i)) + str(accuracy))
        cv_accuracies.append(accuracy)

    print("Mean Accuracy: " + str(np.mean(cv_accuracies)))
    print("STD Accuracy: " + str(np.std(cv_accuracies)))