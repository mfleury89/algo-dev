import numpy as np


def get_index_start_change(arr_net_out, first_index_unstable, max_look_back, threshold_derivative):
    first_index = np.max((first_index_unstable - max_look_back, 0))
    data_uncertain = arr_net_out[:, first_index:first_index_unstable]
    discrete_entropy_derivative = np.diff(-np.sum(data_uncertain * np.log(data_uncertain), axis=0))
    index_transition_start = 0
    for i in range(first_index_unstable - first_index - 1):
        if discrete_entropy_derivative[i] > threshold_derivative:
            index_transition_start = i
            break

    return first_index + index_transition_start


def generate_pseudo_labels(arr_net_out, class_names, unstable_len, threshold_stable,
                           max_len_unstable, max_look_back, threshold_derivative):
    pseudo_labels = []
    pseudo_labels_indices = []
    arr_pred = [class_names[idx] for idx in np.argmax(arr_net_out, axis=0)]
    begin_arr = arr_net_out[:, unstable_len]
    stable = True
    current_class = class_names[np.argmax(np.median(begin_arr, axis=-1))]
    arr_unstable_output = []
    first_index_unstable = 0
    pred_len = len(arr_pred)
    for i in range(pred_len):
        if current_class != arr_pred[i] and stable:
            stable = False
            first_index_unstable = i
            arr_unstable_output = []

        if not stable:
            arr_unstable_output.append(arr_net_out[:, i])
            current_unstable_len = len(arr_unstable_output)

            if current_unstable_len > unstable_len:
                del arr_unstable_output[0]
                current_unstable_len = len(arr_unstable_output)

            if current_unstable_len >= unstable_len:
                arr_median = np.median(np.array(arr_unstable_output).T, axis=-1)
                arr_percentage_medians = arr_median / np.sum(arr_median)
                max_median_index = np.argmax(arr_percentage_medians)
                gesture_found = class_names[max_median_index]

                if arr_percentage_medians[max_median_index] > threshold_stable:
                    stable = True

                    if current_unstable_len < max_len_unstable:
                        for j in range(first_index_unstable, i + 1):
                            pseudo_labels.append(gesture_found)
                            pseudo_labels_indices.append(j)

                        if current_class != gesture_found:
                            index_start_change = get_index_start_change(arr_net_out, first_index_unstable,
                                                                        max_look_back, threshold_derivative)

                            for j in range(index_start_change, first_index_unstable):
                                if j in pseudo_labels_indices:
                                    pseudo_labels[j] = gesture_found

                    current_class = gesture_found
                    arr_unstable_output = []

        else:
            pseudo_labels.append(current_class)
            pseudo_labels_indices.append(i)

    return pseudo_labels, pseudo_labels_indices


if __name__ == '__main__':
    names = ['a', 'b', 'c']
    np.random.seed(42)
    outputs = np.random.rand(len(names), 1000)
    indices = np.random.choice(np.arange(0, 1000), size=500)
    outputs[0, indices] += 1
    outputs /= np.sum(outputs, axis=0)
    print(outputs)
    unstable_length = 100
    threshold_stable_ = 0.3
    max_length_unstable = 200
    max_look_back_ = 50
    threshold_derivative_ = 1

    old_labels = [names[idx] for idx in np.argmax(outputs, axis=0)]
    new_labels, new_indices = generate_pseudo_labels(outputs, names, unstable_length, threshold_stable_,
                                                     max_length_unstable, max_look_back_, threshold_derivative_)

    print(len(old_labels))
    print(old_labels)
    print(len(new_labels))
    print(new_labels)
    relabels = old_labels.copy()

    for i, label in zip(new_indices, new_labels):
        relabels[i] = label

    print(len(relabels))
    print(relabels)
