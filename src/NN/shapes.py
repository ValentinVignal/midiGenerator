import warnings

import src.mtypes as t

'''
def compute_shapes_before_pooling(input_shape: t.shape, model_param_conv: type_model_param_conv) -> t.List[
    t.bshape]:
    """

    :param input_shape:
    :param model_param_conv:
    :return:
    """
    # Create the fake filters : model_param_conv = [[a, b], [c], [d, e]]
    fake_model_param_conv_temp = copy.deepcopy(model_param_conv)
    # -> [[nb_instruments, a, b], [b, c], [c, d, e]] -> [a, b, d]       (shape before pooling)
    fake_model_param_conv_temp[0].insert(0, nb_instruments)
    for i in range(1, len(model_param_conv)):
        fake_model_param_conv_temp[i].insert(0, fake_model_param_conv_temp[i - 1][-1])
    fake_model_param_conv: t.List[int] = [l[-2] for l in fake_model_param_conv_temp]

    new_shapes: t.List[t.shape] = mlayers.conv.new_shapes_conv(
        input_shape=(1, *input_shape[1:-1], fake_model_param_conv[0]),
        strides_list=[(1, time_stride, 2) for i in range(len(model_param_conv) - 1)],
        filters_list=fake_model_param_conv[1:]
    )
    final_shapes = new_shapes[::-1]
    final_bshapes = [t.Bshape.cast_from(shape, t.shape) for shape in final_shapes]
    return final_bshapes


def compute_model_param_dec(input_shape: t.shape,
                            model_param_enc: type_model_param
                            ) -> t.Tuple[type_model_param, t.shape]:
    conv_enc = model_param_enc['conv']
    dense_enc = model_param['dense']

    # --- Compute the last size of the convolution (so we can add a dense layer of this size in the decoder ---
    nb_pool = len(conv_enc) - 1  # nb_times there is a stride == 2 in the conv encoder
    # 1. compute the last shape
    last_shapes_conv_enc = mlayers.conv.new_shapes_conv(input_shape=(*input_shape[:-1], conv_enc[-1][-1]),
                                                        strides_list=[(1, time_stride, 2) for i in
                                                                      range(nb_pool)],
                                                        filters_list=[conv_enc[-1][-1] for i in range(nb_pool)])
    last_shape_conv_enc = last_shapes_conv_enc[-1]
    # 2. compute the last size
    last_size_conv_enc = 1
    for i, s in enumerate(last_shape_conv_enc[1:]):  # Don't take the time axis (1 step only in decoder)
        last_size_conv_enc *= s

    # --- Create the dictionnary to return ---
    model_param_dec = dict(
        dense=dense_enc[::-1] + [last_size_conv_enc],
        conv=mlayers.conv.reverse_conv_param(original_dim=nb_instruments, param_list=conv_enc)
    )
    return model_param_dec, (1, *last_shape_conv_enc[1:])
'''


def time_stride(step_length):
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
