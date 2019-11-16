import warnings
import copy
import math

import src.mtypes as t
import src.NN.layers as mlayers


def compute_shapes_before_pooling(input_shape: t.shape, model_param_conv: t.List[t.List[int]], strides: t.Tuple[int, ...]) -> t.List[t.bshape]:
    """

    :param input_shape:
    :param model_param_conv:
    :param strides:
    :return:
    """
    all_sizes = [
        [None for i in range(len(model_param_conv))]        # Batch dimension
    ]
    # Convolution dimension
    for d in range(len(input_shape) - 1):
        # For all the dimension
        all_sizes.append(
            compute_size_before_pooling(input_shape[d], len(model_param_conv), strides[d])
        )
    # Filter dimension
    all_sizes.append(
        compute_filters_before_pooling(input_shape[-1], model_param_conv)
    )

    shapes = list(zip(*all_sizes))

    return shapes


def compute_size_before_pooling(input_size: int, nb_pool: int, stride: int = 2) -> t.List[int]:
    """

    :param input_size: int
    :param nb_pool:
    :param stride: int

    :return:
    """
    sizes = [input_size]
    for p in range(nb_pool):
        sizes.append(math.ceil(sizes[-1] / stride))
    return sizes[:-1][::-1]


def compute_filters_before_pooling(input_filters: int, filters: t.List[t.List[int]]) -> t.List[int]:
    """

    :param input_filters:
    :param filters:
    :return:
    """
    # Create the fake filters : model_param_conv = [[a, b], [c], [d, e]]
    filters_temp = copy.deepcopy(filters)
    # -> [[nb_instruments, a, b], [b, c], [c, d, e]] -> [a, b, d]       (shape before pooling)
    filters_temp[0].insert(0, input_filters)
    for i in range(1, len(filters)):
        filters_temp[i].insert(0, filters_temp[i - 1][-1])
    filters_final: t.List[int] = [l[-2] for l in filters_temp]
    return filters_final[::-1]


def compute_model_param_dec(input_shape: t.shape,
                            model_param_enc: t.Dict[str, t.Union[t.List[t.List[int]], t.List[int]]],
                            strides: t.List[t.Tuple[int, ...]]
                            ) -> t.Tuple[t.Dict[str, t.Union[t.List[t.List[int]], t.List[int]]], t.shape]:
    """

    :param input_shape: real input of the convolution
    :param model_param_enc:
    :param strides:
    :return:
    """
    conv_enc = model_param_enc['conv']
    dense_enc = model_param_enc['dense']

    # --- Compute the last size of the convolution (so we can add a dense layer of this size in the decoder ---
    nb_pool = len(conv_enc) - 1  # nb_times there is a stride == 2 in the conv encoder
    # 1. compute the last shape
    last_shapes_conv_enc = mlayers.conv.new_shapes_conv(input_shape=(*input_shape[:-1], conv_enc[-1][-1]),
                                                        strides_list=strides,
                                                        filters_list=[conv_enc[-1][-1] for i in range(nb_pool)])
    last_shape_conv_enc = last_shapes_conv_enc[-1]
    # 2. compute the last size
    last_size_conv_enc = 1
    for i, s in enumerate(last_shape_conv_enc[1:]):  # Don't take the time axis (1 step only in decoder)
        last_size_conv_enc *= s

    # --- Create the dictionnary to return ---
    model_param_dec = dict(
        dense=dense_enc[::-1] + [last_size_conv_enc],
        conv=mlayers.conv.reverse_conv_param(original_dim=input_shape[-1], param_list=conv_enc)
    )
    return model_param_dec, (1, *last_shape_conv_enc[1:])

