import numpy as np


def sequence_to_numpy(sequence):
    """

    :param sequence:
    :return:
    """
    x, y = sequence[0]
    x_is_list = isinstance(x, list)     # To know if there is several inputs
    y_is_list = isinstance(y, list)     # To know if there is several outputs
    x_list, y_list = [[x_] for x_ in x], [[y_] for y_ in y]         # Lists to concatenate and return at the end
    if x_is_list:
        nb_x = len(x)       # number of inputs
    if y_is_list:
        nb_y = len(y)       # Number of outputs
    for i in range(1, len(sequence)):
        x, y = sequence[i]
        if x_is_list:
            for j in range(nb_x):
                x_list[j].append(x[j])
        else:
            x_list.append(x)
        if y_is_list:
            for j in range(nb_y):
                y_list[j].append(y[j])
        else:
            y_list.append(y)
    if x_is_list:
        x = [np.concatenate(x_, axis=0) for x_ in x_list]
    else:
        x = np.concatenate(x_list, axis=0)
    if y_is_list:
        y = [np.concatenate(y_, axis=0) for y_ in y_list]
    else:
        y = np.concatenate(y_list, axis=0)

    return x, y








