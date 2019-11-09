def get_shape(t, ax):
    """

    :param t:
    :param ax:
    :return:
    """
    if t[ax] is None or isinstance(t[ax], int):
        return t[ax]
    else:
        return t[ax].value

