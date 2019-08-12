import tensorflow as tf

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda

"""

Just fully connected layers

"""


def create_model(input_param, model_param, nb_steps, optimizer):
    """

    :param input_param:
    :param nb_steps:
    :param model_param:
    :return: the neural network
    """

    print('Definition of the graph ...')

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    midi_shape = (nb_steps, input_size, 2)  # (batch, nb_step, input_size, 2)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size, 2)]

    # ---------- Separated ----------
    first_layer = []
    separated = model_param['separated1']
    for instrument in range(nb_instruments):
        x_a = Lambda(lambda xl: xl[:, :, :, 0])(inputs_midi[instrument])  # activation (batch, nb_steps, input_size)
        x_d = Lambda(lambda xl: xl[:, :, :, 1])(inputs_midi[instrument])  # duration (batch, nb_steps, input_size)

        x = x_a

        for s in separated:
            x = layers.Dense(s)(x)
            x = layers.LeakyReLU()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)

        first_layer.append(x)

    # ---------- Common ----------
    x = layers.concatenate(first_layer, axis=2)
    x = layers.Flatten()(x)
    common = model_param['common']
    for s in common:
        size = s * nb_instruments * nb_steps
        x = layers.Dense(size)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

    # ---------- Separated 2 ----------
    outputs = []        # (batch, nb_steps, nb_instruments, input_size)
    separated = model_param['separated2']
    for instrument in range(nb_instruments):
        for s in separated:
            size = s * nb_instruments * nb_steps
            x = layers.Dense(size)(x)
            x = layers.LeakyReLU()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)

        # ---------- Final Output ----------
        output_a = layers.Dense(input_size, activation='sigmoid')(x)    # (batch, input_size)
        output_a = layers.Reshape((input_size, 1))(output_a)        # (batch, input_size, 1)
        output_d = layers.Dense(input_size, activation='sigmoid')(x)      # (batch, input_size)
        output_d = layers.LeakyReLU()(output_d)
        output_d = layers.Reshape((input_size, 1))(output_d)        # (batch, input_size, 1)
        output = layers.concatenate([output_a, output_d], axis=2)      # (batch, input_size, 2)
        output = layers.Layer(name='Output_{0}'.format(instrument))(output)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    # ------------------ Losses -----------------
    lambda_activation = 20
    lambda_duration = 1

    def custom_loss(lambda_a, lambda_d):

        def loss_function(y_true, y_pred):
            y_true_a = Lambda(lambda x: x[:, :, 0])(y_true)
            y_true_d = Lambda(lambda x: x[:, :, 1])(y_true)
            y_pred_a = Lambda(lambda x: x[:, :, 0])(y_pred)
            y_pred_d = Lambda(lambda x: x[:, :, 1])(y_pred)

            loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
            loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

            loss = lambda_a * loss_a + lambda_d * loss_d

            return loss

        return loss_function

    # Define losses dict
    losses = {}
    for i in range(nb_instruments):
        losses['Output_{0}'.format(i)] = custom_loss(lambda_activation, lambda_duration)

    model.compile(loss=losses, optimizer=optimizer)

    return model, custom_loss(10, 1)
