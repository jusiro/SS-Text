import numpy as np


def standard_split(x, y, p=None, seed=None):
    """
        Perform a standard p/(1-p) split between calibration and validation sets stratified on the classes.
        arguments:
            x [numpy.array]: extracted predictions
            y [numpy.array]: corresponding labels
            p: calibration data proportion
        returns:
            x_calib [numpy.array]: predictions of the calibration set
            y_calib [numpy.array]: calibration set labels
            x_val [numpy.array]: predictions of the validation set
            y_val [numpy.array]: calibration set labels

    """
    calib_idx, val_idx = [], []
    for val in np.unique(y):
        idx = np.where(y == val)[0]
        split_value = np.max([1, int(len(idx)*p)])
        if seed is not None:  # Reproducibility
            np.random.seed(seed)
        np.random.shuffle(idx, )
        calib_idx.extend(idx[:split_value])
        val_idx.extend(idx[split_value:])
    assert set(calib_idx).intersection(set(val_idx)) == set(), 'Overlapping indices.'

    x_calib, y_calib = x[calib_idx], y[calib_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    return x_calib, y_calib, x_val, y_val


def balance_split(x, y, k=16, p=None, seed=None, allow_missing_class=False):

    # Labels as integers
    y = np.int8(y)

    # Total number of samples
    N = len(np.unique(y)) * k

    # Create sampling number
    correct_n = 0
    split_values = []
    for val in list(np.unique(y)):
        split_value = round(N*p[val])
        if not allow_missing_class:
            if np.max(split_value) < 1:
                split_value = np.max([1, split_value])  # + random.sample([0, 1], k=1)[0]
                correct_n += 1
        split_values.append(split_value)

    # Correct N: removing samples from majority class until reaching N
    correct_n = np.sum(split_values) - N
    while correct_n > 0:
        idx = np.argmax(split_values)
        split_values[idx] = split_values[idx] - 1
        correct_n -=1

    # Retrieve indexes
    calib_idx, val_idx = [], []
    for val in list(np.unique(y)):
        idx = np.where(y == val)[0]
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx, )
        calib_idx.extend(idx[:split_values[val]])

    # Create partitions
    x_calib, y_calib = x[calib_idx], y[calib_idx]
    return x_calib, y_calib