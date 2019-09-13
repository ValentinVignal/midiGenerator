import tensorflow as tf

K = tf.keras.backend
layers = tf.keras.layers
import numpy as np


# --------------------------------------------

batch = 32
dim1 = 5
dim2 = 7
dim3 = 11

arange = batch * dim1 * dim2 * dim3
dims = (dim1, dim2, dim3)


class BatchNormalization(layers.Layer):

    def __init__(self, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, inputs_shape):
        self.zeros = self.add_weight(name='batch_norm_pop_mean',
                                     shape=dims,
                                     initializer=tf.keras.initializers.zeros,
                                     trainable=False)
        self.ones = self.add_weight(name='batch_norm_pop_var',
                                    shape=dims,
                                    initializer=tf.keras.initializers.ones,
                                    trainable=False)

    def call(self, inputs, training=None):
        def batch_norm_train():
            return tf.multiply(self.ones, inputs)

        def batch_norm_no_train():
            return tf.multiply(self.zeros, inputs)

        return K.in_train_phase(batch_norm_train, batch_norm_no_train, training=training)

        #if not K.eval(K.learning_phase()):
        #    return batch_norm_train()
        #else:
        #    return batch_norm_no_train()

    def compute_output_shape(self, input_shape):
        return input_shape


def add(ip):
    a = ip[0]
    b = ip[1]
    return a * b


input_model1 = layers.Input((dim1, dim2, dim3))
input_model2 = layers.Input((dim1, dim2, dim3))
# input_learning = layers.Input((1,))
x1 = BatchNormalization()(input_model1)
x = layers.Lambda(add)([x1, input_model2])
x1 = layers.Layer()(x)
x2 = layers.Lambda(lambda xl: 2 * xl)(x)
om1 = layers.Reshape((dim1, dim2, dim3, 1))(x1)
om2 = layers.Reshape((dim1, dim2, dim3, 1))(x2)

model = tf.keras.Model(inputs=[input_model1, input_model2], outputs=[om1, om2])

i1 = np.reshape(np.arange(arange), (batch, *dims))
i2 = np.reshape(np.arange(arange), (batch, *dims))

rp = np.zeros((batch, *dims))
rt = np.reshape(i1 * i2, (batch, *dims, 1))


print('normal predict')
result = model.predict([i1, i2])
print(result)


optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='mae', metrics=['acc', 'mae', tf.keras.metrics.categorical_accuracy], optimizer=optimizer)
model.fit([i1, i2], [rt, 2 * rt])
print('\npredict_function')
model._make_predict_function()
print(model.predict_function([i1, i2, 1]))

print('\ntest function')
model._make_test_function()
ins = [i1, i2] + [rt, 2*rt] + [np.ones((1, )) for e in range(2)] + [0]
print(model.test_function(ins))

# --------------------------------------------


"""
f = K.function([input_model, K.learning_phase()], [output_model])

result = f([np.reshape(np.arange(8), (2, 4)), 0])
print(result)
"""


"""
result = model.predict([np.arange(4), 0], verbose=1)
print('result', result)
"""


"""
K.set_learning_phase(1)
model.fit(x=np_inputs, y=np_outputs, epochs=10)

K.set_learning_phase(0)
print('\n\nevaluation')
evaluation = model.evaluate(x=np_inputs, y=np_outputs)
print('eval:', evaluation)
"""

