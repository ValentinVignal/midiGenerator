def eval_all(x, env):
    """

    :param x: return x, if x is str, return eval(x)
    :param env: dic of environment variables
    :return:
    """
    if type(x) is str:
        return eval(x, env)
    else:
        return x


def eval_object(x, env):
    """

    :param x:
    :param env:
    :return:
    """
    if isinstance(x, dict):
        for key in x:
            x[key] = eval_object(x[key], env)
    elif isinstance(x, list):
        for index, value in enumerate(x):
            x[index] = eval_object(value, env)
    elif isinstance(x, str):
        return eval(x, env)
    else:
        pass
    return x





