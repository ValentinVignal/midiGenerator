import tensorflow as tf

import src.eval_string as es

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda

"""

First personal model

"""




def create_model(input_param, model_param, nb_steps, optimizer):
    """

    :param input_param:
    :param nb_steps:
    :param model_param:
    :return: the neural network
    """

    # ---------- Functions for DenseNet ----------
    def denseBlock(k, nb_conv):
        """

        :param k:
        :param nb_conv:
        :return:
        """
        def f(input):
            output = input
            for conv in range(nb_conv):
                o = layers.BatchNormalization()(output)
                o = layers.ReLu()(o)
                o = layers.Conv2D(filters=k, kernel_size=(3, 3), padding='same')(o)
                o = layers.Dropout(0.2)(o)
                output = layers.concatenate([output, o], axis=3)

            return output
        return f

    def denseTransitionBlock(k):
        """

        :param k:
        :return:
        """
        def f(input):
            output = layers.BatchNormalization()(input)
            output = layers.RELU()(output)
            output = layers.Conv2D(filters=k,
                                   kernel_size=(1, 1),
                                   padding='same')(output)
            output = layers.Dropout(0.2)(output)
            output = layers.MaxPool2D(pool_size=(3, 3),
                                      stride=(2, 2),
                                      padding='same')(output)
            return output
        return f

    # ---------- Neural Network ----------
    print('Definition of the graph ...')

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    midi_shape = (nb_steps, input_size, 2)  # (batch, nb_step, input_size, 2)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size, 2)]

    # ---------- Separated network for instruments ----------
    # Nothing to do

    # ---------- Concatenation ----------
    x = layers.concatenate(inputs_midi, axis=3)     # (batch, nb_steps, input_size, 2 * nb_instruments)

    # ---------- All together ----------
    dense_param = model_param['dense_param']
    x = denseBlock(k=dense_param['k'][0],
                   nb_conv=dense_param['nb_conv'][0])()
    for i in range(1, len(dense_param['k'])):
        x = denseTransitionBlock(k=dense_param['k'][i])
        x = denseBlock(
            k=dense_param['k'][i],
            nb_conv=dense_param['nb_conv'][i])(x)

    x = layers.Flatten()(x)

    for i in range(len(model_param['fc_common'])):
        x = layers.Dense(es.eval_all(common['fc'][i], env=env))(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.4)(x)

    # ---------- Instruments separately ----------
    separated2 = model_param['separated2']
    outputs = []        # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        o = x
        for i in range(len(separated2['fc'])):
            o = layers.Dense(es.eval_all(separated2['fc'][i], env=env))(o)
            o = layers.LeakyReLU()(o)
            o = layers.BatchNormalization()(o)
            o = layers.Dropout(0.3)(o)
        output_a = layers.Dense(input_size, activation='sigmoid')(o)    # (batch, input_size)
        output_a = layers.Reshape((input_size, 1))(output_a)        # (batch, input_size, 1)
        output_d = layers.Dense(input_size, activation='sigmoid')(o)      # (batch, input_size)
        output_d = layers.LeakyReLU()(output_d)
        output_d = layers.Reshape((input_size, 1))(output_d)        # (batch, input_size, 1)
        output = layers.concatenate([output_a, output_d], axis=2)      # (batch, input_size, 2)
        output = layers.Layer(name='Output_{0}'.format(instrument))(output)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    # ------------------ Loss -----------------

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

    model.compile(loss=custom_loss(10, 1), optimizer=optimizer)

    return model, custom_loss(10, 1)
