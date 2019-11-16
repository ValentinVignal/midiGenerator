import warnings

import src.mtypes as t


def time_stride(step_length: int) -> int:
    """

    :param step_length:
    :return:
    """
    if step_length == 4 or step_length == 1:
        # It is working either on 'Beat' or on 'Note'
        # We don't want to stride on the time axis
        ts = 1
    elif step_length == 16:
        # It is working on 'Measure'
        # We want to stride on the time axis (the size is considered as big enough)
        ts = 2
    else:
        warnings.warn(
            f'The model is not designed wo work with a step_length {step_length} not included in (8, 16),' +
            'some errors might occur',
            Warning)
        ts = 1 if step_length < 16 else 2
    return ts


def time_step_to_x(l: t.List[t.bshape], axis: int = 1, x: int = 1) -> t.List[t.bshape]:
    """

    :param l:
    :param axis:
    :param x:

    :return:
    """
    for i in range(len(l)):
        t = list(l[i])
        t[axis] = x
        l[i] = tuple(t)
    return l


