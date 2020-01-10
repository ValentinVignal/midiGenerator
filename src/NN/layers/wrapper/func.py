def apply_same_on_list(layer):
    """

    :param layer:
    :return:
    """
    def _apply_same_on_list(x):
        return [layer(x_) for x_ in x]
    return _apply_same_on_list


def apply_different_on_list(layers):
    """

    :param layers:
    :return:
    """
    def _apply_different_on_list(x):
        if len(layers) != len(x):
            raise IndexError(f'Number of inputs {len(x)} != number of layers {len(layers)}')
        return [layers[i](x[i]) for i in range(len(layers))]
    return _apply_different_on_list


def apply_different_layers(layers):
    """

    :param layers:
    :return:
    """
    def _apply_different_layers(x):
        return [layer(x) for layer in layers]
    return _apply_different_layers
