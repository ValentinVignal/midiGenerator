import tensorflow as tf
import warnings

import src.eval_string as es
import src.NN.losses as l
import src.global_variables as g

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda
K = tf.keras.backend

"""

LSTM with encoder convolutional layers and the deconder use transposed convolutional layer

"""


def create_model(input_param, model_param, nb_steps, step_length, optimizer, type_loss=g.type_loss,
                 model_options={}
                 ):
    """

    :param input_param:
    :param nb_steps:
    :param model_param:
    :param step_length:
    :param optimizer:
    :param type_loss:
    :param model_options:
    :return: the neural network:
    """

    # ---------- Model options ----------
    mmodel_options = {
        'dropout': g.dropout,
        'all_sequence': g.all_sequence,
        'lstm_state': g.lstm_state,
        'no_batch_norm': False,
        'bn_momentum': g.bn_momentum,
        'lambdas_loss': g.lambdas_loss,
        'last_fc': g.last_fc
    }
    mmodel_options.update(model_options)

    dropout = mmodel_options['dropout']
    all_sequence = mmodel_options['all_sequence']
    lstm_state = mmodel_options['lstm_state']
    batch_norm = not mmodel_options['no_batch_norm']
    bn_momentum = mmodel_options['bn_momentum']
    last_fc = mmodel_options['last_fc']

    lambda_loss_activation, lambda_loss_duration = g.get_lambdas_loss(mmodel_options['lambdas_loss'])
    # --------- End model options ----------

    print('Definition of the graph ...')

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    if step_length == 8 or step_length == 1:
        # It is working either on 'Beat' or on 'Note'
        # We don't want to stride on the time axis
        time_stride = 1
    elif step_length == 16:
        # It is working on 'Measure'
        # We want to stride on the time axis (the size is considered as big enough)
        time_stride = 2
    else:
        warnings.warn(
            f'The model is not designed wo work with a step+length {step_length}' + 'not included in (8, 16),' +
            'some errors might occur',
            Warning)
        time_stride = 1 if step_length < 16 else 2

    midi_shape = (nb_steps, step_length, input_size, 2)  # (batch, step_length, nb_step, input_size, 2)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size, 2)]

    # ---------- All together ----------
    x = layers.concatenate(inputs_midi, axis=4)  # (batch, nb_steps, step_length, input_size, nb_instruments)

    # ================================================================================
    #                                  Encoder
    # ================================================================================

    # --------------------------------------------------
    # ----- Convolutional Layer -----
    # --------------------------------------------------
    convo_shapes = []  # to reconstruct the good shape in Decoder and Transposed convolutions
    convo = model_param['convo']
    for i, c in enumerate(convo):
        convo_shapes.append(x.shape)
        for index, s in enumerate(c):
            size = s
            if index + 1 == len(c) and i < 2:
                x = layers.Conv3D(filters=size, kernel_size=(1, 5, 5), strides=(1, time_stride, 2), padding='same')(x)
            else:
                x = layers.Conv3D(filters=size, kernel_size=(1, 5, 5), padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization(momentum=bn_momentum)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(dropout / 2)(x)
        # x = layers.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 1, 2), padding='same')(x)

    # --------------------------------------------------
    # ---------- Fully Connected ----------
    # --------------------------------------------------
    shape_before_fc = x.shape  # Do  the bridge between Fully Connected Transposed Convolution in the Decoder
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3] * x.shape[4]))(
        x)  # (batch, nb_steps, lenght * size * filters)
    fc = model_param['fc']
    for s in fc:
        size = eval(s, env)
        x = layers.TimeDistributed(layers.Dense(size))(x)
        if batch_norm:
            x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)

    # ================================================================================
    #                         Recurrent Neural Network
    # ================================================================================

    # --------------------------------------------------
    # ---------- LSTM -----------
    # --------------------------------------------------
    # -- Loop --
    lstm = model_param['LSTM']
    for s in lstm[:-1]:
        size = eval(s, env)
        x = layers.LSTM(size,
                        return_sequences=True,
                        unit_forget_bias=True,
                        dropout=dropout,
                        recurrent_dropout=dropout)(x)  # (batch, nb_steps, size)
        if batch_norm:
            x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)
    # -- Last one --
    if lstm_state:
        s = lstm[-1]
        size = eval(s, env)
        x, state_h, state_c = layers.LSTM(size,
                                          return_sequences=all_sequence,
                                          return_state=True,
                                          unit_forget_bias=True,
                                          dropout=dropout,
                                          recurrent_dropout=dropout)(x)  # (batch, nb_steps, size)
        if batch_norm:
            x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)
        if all_sequence:
            x = layers.Flatten()(x)

        x = layers.concatenate([x, state_h, state_c], axis=1)  # (batch, 3 *  size)
    else:
        s = lstm[-1]
        size = eval(s, env)
        x = layers.LSTM(size,
                        return_sequences=all_sequence,
                        return_state=False,
                        unit_forget_bias=True,
                        dropout=dropout,
                        recurrent_dropout=dropout)(x)  # (batch, nb_steps, size)
        if batch_norm:
            x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.LeakyReLU()(x)
        if all_sequence:
            x = layers.Flatten()(x)
        x = layers.Dropout(dropout)(x)

    # ================================================================================
    #                                  Decoder
    # ================================================================================

    # --------------------------------------------------
    # ----- Fully Connected ------
    # --------------------------------------------------
    fc_decoder = model_param['fc'][::-1] + [
        shape_before_fc[2] * shape_before_fc[3] * shape_before_fc[4]]  # Do the fully connected but backward
    """
    We keep the last layer (so now the first one in fc_decoder) because the output shape of the lstm is the number of
    units in the last LSMT layer (and possibly * le length of the all sequence + size of the state) != last dimension
    of the fully connected layers.
    
    We add another layer at the end of fc_decoder to have the same size as the last layer of the flatten encoder
    convolution.
    """
    for s in fc_decoder:
        size = es.eval_all(s, env)
        x = layers.Dense(size)(x)
        if batch_norm:
            x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)  # (batch, size)

    # --------------------------------------------------
    # ----- Transposed Convolution -----
    # --------------------------------------------------
    x = layers.Reshape((1, shape_before_fc[2], shape_before_fc[3], shape_before_fc[4]))(
        x)  # (batch, nb_steps=1, length, size, filters)
    # Create the parameters transposed_convo
    """
    (nb_instrument * 2 ->) [[a, b], [c, d, e]] has to become (e ->) [[d, c, b], [a, nb_instruments]]
    And the UpSampling is done on the first convolution
    
    To do so:
        (1)     [[a, b]] , [c, d, e]]       <-- model_param['convo']
        (2)     [a, b, c, d, e]       # dims = [2, 3]
        (3)     [e, d, c, b, a]       # dims = [3, 2]
        (3)     [d, c, b, a , 2 * nb_instruments]       # save dims = [3, 2]
        (4)     [[d, c, b], [a, 2 * nb_instuments]]     <-- transposed_convo
    """
    transposed_convo_dims = [len(sublist) for sublist in model_param['convo']]
    transposed_convo_temp = [size for sublist in model_param['convo'] for size in
                             sublist]  # Flatten the 2-level list
    transposed_convo_temp = transposed_convo_temp[::-1]  # Reversed
    transposed_convo_dims = transposed_convo_dims[::-1]
    transposed_convo_temp = transposed_convo_temp[1:] + [2 * nb_instruments]  # Update shapes
    transposed_convo = []  # Final transposed_convo parameters
    offset = 0
    for sublist_size in transposed_convo_dims:
        transposed_convo.append(transposed_convo_temp[offset: offset + sublist_size])
        offset += sublist_size

    for tc_idx, tc in enumerate(transposed_convo):
        for s_idx, s in enumerate(tc):
            if s_idx == 0 and tc_idx > 0:
                strides = (1, time_stride, 2)
            else:
                strides = (1, 1, 1)
            size = s
            x = layers.Conv3DTranspose(filters=size, kernel_size=(1, 5, 5), padding='same', strides=strides)(
                x)  # (batch, nb_step=1, step_size, input_size, filters)
            if s_idx == 0 and tc_idx > 0:
                if x.shape[3] != convo_shapes[- (tc_idx + 1)][3]:
                    # Correction of the input size (if it was an odd number, we need to -1)
                    x = layers.Lambda(lambda x: x[:, :, :, :-1])(x)
                if x.shape[2] != convo_shapes[- (tc_idx + 1)][2]:
                    # Correction of the step_length (if it was an odd number, we need to -1)
                    x = layers.Lambda(lambda x: x[:, :, :-1])(x)
            if batch_norm:
                x = layers.BatchNormalization(momentum=bn_momentum)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(dropout / 2)(x)
        # x = layers.UpSampling3D(size=(1, 1, 2))(x)  # Batch size

    # Delete the dimension nb_steps = 1
    x = layers.Reshape((x.shape[2:]))(x)  # (batch, step_size, input_size, 2 * nb_instruments)

    if last_fc:
        """
            If we want to add an extra fully connected at the end to "summarize"
        """
        # --------------------------------------------------
        # ---------- Instruments separately ----------
        # --------------------------------------------------
        x = layers.Flatten()(x)
        x = layers.Dense((step_length * input_size * (2 * nb_instruments)))(x)
        x = layers.Reshape((step_length, input_size, 2 * nb_instruments))(x)

    # x : (batch, step_size, input_size, 2 * nb_instruments)

    x = layers.Lambda(lambda x_: tf.keras.activations.sigmoid(x_), name='Last_Activation')(x)  # Might not work

    outputs = []
    for inst in range(nb_instruments):
        outputs.append(layers.Lambda(lambda x_: x_[:, :, :, inst:inst + 2], name=f'Output_{inst}')(x))

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    # ------------------ Losses -----------------
    # Define losses dict
    losses = {}
    for inst in range(nb_instruments):
        losses[f'Output_{inst}'] = l.choose_loss(type_loss)(lambda_loss_activation, lambda_loss_duration)

    model.compile(loss=losses, optimizer=optimizer, metrics=[l.acc_act, l.mae_dur])

    return model, losses, (lambda_loss_activation, lambda_loss_duration)
