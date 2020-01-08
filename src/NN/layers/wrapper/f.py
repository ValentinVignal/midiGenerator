def apply_same_on_list(layer, x):
    return [layer(x_) for x_ in x]


def apply_different_on_list(layers, x):
    if len(layers) != len(x):
        raise IndexError(f'Number of inputs {len(x)} != number of layers {len(layers)}')
    return [layers[i](x[i]) for i in range(len(layers))]


def apply_different_layers(layers, x):
    return [layer(x) for layer in layers]




