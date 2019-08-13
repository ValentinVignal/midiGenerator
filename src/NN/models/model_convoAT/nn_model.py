import tensorflow as tf

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda

"""

Convolution all together

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

    # ---------- Convolutions ----------
    x = layers.concatenate(inputs_midi, axis=3)  # (batch, nb_steps, input_size, 2 * nb_instruments)

    convo = model_param['convo']
    for i in range(len(convo)):
        for j in range(len(convo[i])):
            x = layers.Conv2D(filters=nb_instruments * convo[i][j],
                              kernel_size=(3, 3),
                              padding='same')(x)
            x = layers.LeakyReLU()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # ---------- Fully Connected ----------
    x = layers.Flatten()(x)

    fc = model_param['fc']
    for i in range(len(fc)):
        x = layers.Dense(fc[i] * nb_instruments)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.4)(x)

    # ---------- Final Output ----------
    outputs = []        # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
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

    return model, custom_loss(lambda_activation, lambda_duration)
