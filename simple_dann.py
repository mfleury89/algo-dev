import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


def train_simple_dann(labeled_samples, labels, unlabeled_samples, n_nodes=15, learning_rate=0.01,
                      adaptation_parameter=6, max_iterations=1000, disable_adversity=False, shuffle_unlabeled=True):

    input_dim = labeled_samples.shape[0]
    unique_labels = np.unique(labels)
    n_labels = np.size(unique_labels)

    w = np.random.uniform(size=(n_nodes, input_dim))
    v = np.random.uniform(size=(n_labels, n_nodes))
    b = np.zeros((n_nodes, 1))
    c = np.zeros((n_labels, 1))
    u = np.zeros((n_nodes, 1))
    d = 0

    mapping = {}
    for i, label in enumerate(unique_labels):
        mapping[label] = i

    numerical_labels = np.array([mapping[label] for label in labels])
    one_hot_labels = np.zeros((np.size(labels), n_labels))
    for i, label in enumerate(numerical_labels):
        one_hot_labels[i, label] = 1

    iteration = 0
    while iteration < max_iterations:
        for idx in range(labeled_samples.shape[-1]):
            sample = labeled_samples[:, idx].reshape(input_dim, 1)

            # Forward propagation
            z = sigmoid(np.dot(w, sample) + b)
            probs = softmax(np.dot(v, z) + c)

            # Backpropagation
            delta_c = -(one_hot_labels[idx, :].reshape(n_labels, 1) - probs)
            delta_v = np.dot(delta_c, z.T)
            delta_b = np.multiply(np.dot(v.T, delta_c), np.multiply(z, 1 - z))
            delta_w = np.dot(delta_b, sample.T)

            # Domain adaptation regularizer...
            # ...from current domain
            probs_d = sigmoid(np.dot(u.T, z) + d)
            delta_d = adaptation_parameter * (1 - probs_d)
            delta_u = adaptation_parameter * (1 - probs_d) * z
            if not disable_adversity:
                tmp = adaptation_parameter * (1 - probs_d) * np.multiply(u, np.multiply(z, 1 - z))
                delta_b += tmp
                delta_w += np.dot(tmp, sample.T)

            #  ...from other domain
            if shuffle_unlabeled:
                jdx = np.random.randint(0, unlabeled_samples.shape[-1])  # shuffling if make_moons does not
            else:
                jdx = idx  # make_moons already shuffles data by default
            unlabeled_sample = unlabeled_samples[:, jdx].reshape(input_dim, 1)
            z_u = sigmoid(np.dot(w, unlabeled_sample) + b)
            probs_u = sigmoid(np.dot(u.T, z_u) + d)
            delta_d -= adaptation_parameter * probs_u
            delta_u -= adaptation_parameter * probs_u * z_u
            if not disable_adversity:
                tmp = -adaptation_parameter * probs_u * np.multiply(u, np.multiply(z_u, 1 - z_u))
                delta_b += tmp
                delta_w += np.dot(tmp, unlabeled_sample.T)

            # Update neural network parameters
            w -= learning_rate * delta_w
            v -= learning_rate * delta_v
            b -= learning_rate * delta_b
            c -= learning_rate * delta_c

            # Update domain classifier
            u += learning_rate * delta_u
            d += learning_rate * delta_d

        iteration += 1

    return w, v, b, c


def predict(samples, w, v, b, c):
    return np.argmax(softmax(np.dot(v, sigmoid(np.dot(w, samples) + b)) + c), axis=0)


def compute_accuracy(predictions, labels):
    return np.sum(np.where(predictions == labels, 1, 0)) / np.size(labels)


if __name__ == '__main__':
    labeled_samples_, labels_ = make_moons(n_samples=300, noise=0.1, random_state=42)
    labeled_samples_ = labeled_samples_.T

    angle = 35 * np.pi / 180
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    unlabeled_samples_ = np.dot(rotation_matrix, labeled_samples_)

    # train_indices = []  # shuffling if make_moons does not shuffle data
    # for _ in range(np.size(labels_)//2):
    #     while True:
    #         index = np.random.randint(0, np.size(labels_))
    #         if index not in train_indices:
    #             train_indices.append(index)
    #             break
    #
    # train_indices = np.sort(train_indices)
    # test_indices = np.array(list(set(range(np.size(labels_))).difference(set(train_indices))))

    train_indices = np.array(list(range(np.size(labels_)//2)))  # make_moons already shuffles data by default
    test_indices = np.array(list(range(np.size(labels_)//2, np.size(labels_))))

    plt.scatter(labeled_samples_.T[:, 0], labeled_samples_.T[:, 1], c='g')
    plt.scatter(unlabeled_samples_.T[:, 0], unlabeled_samples_.T[:, 1], c='r')
    plt.show()

    learning_rate_ = 0.001
    max_iterations_ = 10000

    w_, v_, b_, c_ = train_simple_dann(labeled_samples_[:, train_indices], labels_[train_indices],
                                       unlabeled_samples_[:, train_indices], n_nodes=15,
                                       learning_rate=learning_rate_, adaptation_parameter=6,
                                       max_iterations=max_iterations_, disable_adversity=False,
                                       shuffle_unlabeled=False)

    labeled_predictions = predict(labeled_samples_[:, test_indices], w_, v_, b_, c_)
    unlabeled_predictions = predict(unlabeled_samples_[:, test_indices], w_, v_, b_, c_)

    labeled_accuracy = compute_accuracy(labeled_predictions, labels_[test_indices])
    unlabeled_accuracy = compute_accuracy(unlabeled_predictions, labels_[test_indices])

    print(labeled_accuracy)
    print(unlabeled_accuracy)

    w_, v_, b_, c_ = train_simple_dann(labeled_samples_[:, train_indices], labels_[train_indices],
                                       unlabeled_samples_[:, train_indices], n_nodes=15,
                                       learning_rate=learning_rate_, adaptation_parameter=6,
                                       max_iterations=max_iterations_, disable_adversity=True,
                                       shuffle_unlabeled=False)

    labeled_predictions = predict(labeled_samples_[:, test_indices], w_, v_, b_, c_)
    unlabeled_predictions = predict(unlabeled_samples_[:, test_indices], w_, v_, b_, c_)

    labeled_accuracy = compute_accuracy(labeled_predictions, labels_[test_indices])
    unlabeled_accuracy = compute_accuracy(unlabeled_predictions, labels_[test_indices])

    print(labeled_accuracy)
    print(unlabeled_accuracy)
