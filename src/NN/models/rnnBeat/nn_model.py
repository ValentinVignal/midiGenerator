import tensorflow as tf

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
        'lambdas_loss': g.lambdas_loss
    }
    mmodel_options.update(model_options)

    dropout = mmodel_options['dropout']
    all_sequence = mmodel_options['all_sequence']
    lstm_state = mmodel_options['lstm_state']
    batch_norm = not mmodel_options['no_batch_norm']
    bn_momentum = mmodel_options['bn_momentum']

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

    midi_shape = (nb_steps, step_length, input_size, 2)  # (batch, step_length, nb_step, input_size, 2)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size, 2)]

    # ---------- All together ----------
    x = layers.concatenate(inputs_midi, axis=4)  # (batch, nb_steps, step_length, input_size, nb_instruments)

    # ----- Convolutional Layer -----
    convo = model_param['convo']
    for i in convo:
        for index, s in enumerate(i):
            size = s
            if index + 1 == len(i):
                x = layers.Conv3D(filters=size, kernel_size=(1, 5, 5), strides=(1, 1, 2), padding='same')(x)
            else:
                x = layers.Conv3D(filters=size, kernel_size=(1, 5, 5), padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization(momentum=bn_momentum)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(dropout / 2)(x)
        # x = layers.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 1, 2), padding='same')(x)
    shape_before_fc = x.shape
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
    # ---------- LSTM -----------
    size_before_lstm = x.shape[2]  # (batch, nb_steps, size)
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
            x =layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.LeakyReLU()(x)
        if all_sequence:
            x = layers.Flatten()(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(size_before_lstm)(x)  # (batch, size)

    # ----- Fully Connected ------
    fc_decoder = model_param['fc'][::-1]
    for s in (fc_decoder[1:] + [shape_before_fc[2] * shape_before_fc[3] * shape_before_fc[4]]):
        size = es.eval_all(s, env)
        x = layers.Dense(size)(x)
        if batch_norm:
            x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)  # (batch, size)

    # ----- Transposed Convolution -----
    x = layers.Reshape((1, shape_before_fc[2], shape_before_fc[3], shape_before_fc[4]))(
        x)  # (batch, length, size, filters)
    transposed_convo = model_param['convo'][::-1]
    for i in range(len(transposed_convo)):  # inverse the lists inside
        transposed_convo[i] = transposed_convo[i][::-1]
    for i in range(len(transposed_convo) - 1):
        transposed_convo[i] = transposed_convo[i][1:] + [transposed_convo[i + 1][0]]
    transposed_convo[-1] = transposed_convo[-1][1:] + [model_param['last_convo']]
    for tc in transposed_convo:
        for s_idx, s in enumerate(tc):
            strides = (1, 1, 1) if s_idx < len(tc) - 1 else (1, 1, 2)
            size = s
            x = layers.Conv3DTranspose(filters=size, kernel_size=(1, 5, 3), padding='same', strides=strides)(x)
            if batch_norm:
                x = layers.BatchNormalization(momentum=bn_momentum)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(dropout / 2)(x)
        x = layers.UpSampling3D(size=(1, 1, 2))(x)  # Batch size

    x = layers.Flatten()(x)

    # ---------- Instruments separately ----------
    outputs = []  # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        o = x
        output_a = layers.Dense(step_length * input_size, activation='sigmoid')(o)  # (batch, input_size)
        output_a = layers.Reshape((step_length, input_size, 1))(output_a)  # (batch, input_size, 1)
        output_d = layers.Dense(step_length * input_size, activation='sigmoid')(o)  # (batch, input_size)
        output_d = layers.Reshape((step_length, input_size, 1))(output_d)  # (batch, input_size, 1)
        output = layers.concatenate([output_a, output_d], axis=3)  # (batch, input_size, 2)
        output = layers.Layer(name='Output_{0}'.format(instrument))(output)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    # ------------------ Losses -----------------
    # Define losses dict
    losses = {}
    for i in range(nb_instruments):
        losses['Output_{0}'.format(i)] = l.choose_loss(type_loss)(lambda_loss_activation, lambda_loss_duration)

    model.compile(loss=losses, optimizer=optimizer, metrics=[l.acc_act, l.mae_dur])

    return model, losses, (lambda_loss_activation, lambda_loss_duration)