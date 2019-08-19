import tensorflow as tf

Lambda = tf.keras.layers.Lambda


def custom_loss(lambda_a, lambda_d):
    def loss_function(y_true, y_pred):
        y_true_a = Lambda(lambda x: x[:, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, 0])(y_pred)
        y_pred_a_rounded = tf.round(y_pred_a)
        y_pred_d = Lambda(lambda x: x[:, :, 1])(y_pred)

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_a_rounded = tf.keras.losses.mean_squared_error(y_true_a, y_pred_a_rounded)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * (loss_a + loss_a_rounded) + lambda_d * loss_d

        return loss

    return loss_function


def acc_act(y_true, y_pred):
    y_true_a = Lambda(lambda x: x[:, :, 0])(y_true)
    y_pred_a = Lambda(lambda x: x[:, :, 0])(y_pred)

    acc = tf.keras.metrics.binary_accuracy(y_true_a, y_pred_a)

    return acc

def mae_dur(y_true, y_pred):
    y_true_d = Lambda(lambda x: x[:, :, 1])(y_true)
    y_pred_d = Lambda(lambda x: x[:, :, 1])(y_pred)

    mae = tf.keras.metrics.mean_squared_error(y_true_d, y_pred_d)

    return mae
