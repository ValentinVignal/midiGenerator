import numpy as np


def string_to_list(string):
    """
    Create a list from the tupple
    '0:10:2' -> [0, 2, 4, 6, 8]

    :param string:
    :return:
    """
    string_list = string.split(':')
    if len(string_list) == 1:
        return [float(string_list[0])]
    else:
        return list(
            np.arange(float(string_list[0]), float(string_list[1]) + float(string_list[2]), float(string_list[2])))


def string_to_tuple(string, t=float, separator=':'):
    """
    Get the tuple from the string given by the user
    :param separator: separator of the value in the string
    :param t: float, int or other type (function)
    :param string: the string given by the user
    :return:
    """
    return tuple(map(t, string.split(separator)))


def string_to_bool(string):
    """
    Used to evaluate the boolean written in the string
    :param string:
    :return:
    """
    return string == 'True'


def ten_power(x):
    """

    :param x:
    :return: 10 ** (-x)
    """
    return 10 ** (-float(x))
