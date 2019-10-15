import numpy as np
import tensorflow as tf

K = tf.keras.backend
layers = tf.keras.layers

# mask = 1 -> f(x) = 3x + 2
# mask = 0 -> f(x) = -2x + 3


inputs = layers.Input(shape=(1,))
masks = layers.Input((1,))

d0 = layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')
d1 = layers.Dense(1)


class ChooseLayer(layers.Layer):
    def __init__(self):
        self.d0 = layers.Dense(units=1, bias_initializer='zeros', kernel_initializer='zeros')
        self.d1 = layers.Dense(units=1, bias_initializer='ones', kernel_initializer='ones')
        super(ChooseLayer, self).__init__()

    def build(self, input_shape):
        print('shape', input_shape)
        self.d0.build(input_shape[0])
        self.d1.build(input_shape[0])
        self._trainable_weights = []
        self._trainable_weights += self.d0.trainable_weights
        self._trainable_weights += self.d1.trainable_weights
        super(ChooseLayer, self).build(input_shape[0])

    def call(self, inputs):
        i, m = inputs
        return K.switch(m,
                        self.d0(i),
                        self.d1(i))


def choose(i, m):
    o = K.switch(K.equal(m, K.zeros_like(m)),
                 d0(i),
                 d1(i))
    return o


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.l = ChooseLayer()

    def call(self, inputs):
        output = self.l(inputs)
        return output


model = MyModel()
optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mae')
model.build([(None, 1), (None, 1)])             # Build has to be after compile
print('model', model.summary())

# -----------------------





# -------------- Data ---------------

x_np = np.arange(50)
m_np = np.zeros(50)


def f0(x):
    return - 2 * x + 3


def f1(x):
    return 3 * x + 2


m_np[1::2] = 1
y_np = np.copy(x_np)
y_np[::2] = f0(y_np[::2])
y_np[1::2] = f1(y_np[1::2])

m_np = m_np[..., np.newaxis] == 0
x_np = x_np[..., np.newaxis]
y_np = y_np[..., np.newaxis]

print('before', np.concatenate([x_np, y_np], axis=1))

batch_size = 1
print('output names', model.output_names)
model.fit(x=[x_np, m_np], y=y_np, epochs=10, batch_size=batch_size)
y_p = model.predict(x=[x_np, m_np], batch_size=batch_size)

print('after', np.concatenate([x_np, y_np, y_p], axis=1))
print('model', model.summary())


