import numpy as np


def indices_to_flattened(indices, shape):
    # e.g., for (1, 20, 80, 2), i = first*2*80*20 + rows*2*80 + columns*2 + depths
    flattened_index = 0
    for i in range(1, np.size(shape)):
        flattened_index += indices[i - 1] * np.prod(shape[i:])
    flattened_index += indices[-1]
    return flattened_index


def flattened_to_indices(index, shape):
    # e.g., for (1, 20, 80, 2), i // 2*80*20 = d1, (i % 2*80*20) // 2*80 = d2,
    # ((i % 2*80*20) % 2*80) // 2 = d3, ((i % 2*80*20) % 2*80) % 2 == d4
    indices = []
    elements_remaining = index
    for i in range(1, np.size(shape)):
        dim_prod = np.prod(shape[i:])
        indices.append(elements_remaining // dim_prod)
        elements_remaining = elements_remaining % dim_prod
    indices.append(elements_remaining)
    return indices
