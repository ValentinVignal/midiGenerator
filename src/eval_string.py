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
