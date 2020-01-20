def set_default(d, default_dict):
    """
    set the default parameters of default_dict in d

    :param d:
    :param default_dict:
    :return:
    """
    for k in default_dict:
        if not k in d:
            d[k] = default_dict[k]


def set_default_rec(d, default_dict):
    """
    set the default parameters of default_dict in d
    Work recursively if a key of default dict is a dictionary too

    :param d:
    :param default_dict:
    :return:
    """
    raise NotImplementedError



