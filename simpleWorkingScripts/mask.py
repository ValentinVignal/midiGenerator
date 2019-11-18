import numpy as np
import tensorflow as tf
from termcolor import cprint

K = tf.keras.backend
layers = tf.keras.layers

# mask = 1 -> f(x) = 3x + 2
# mask = 0 -> f(x) = -2x + 3

cprint('test', 'blue')

inputs = [layers.Input(shape=(1,)) for i in range(2)]
c = layers.concatenate(inputs, axis=1)
x = layers.Dense(1)(c)
masks = layers.Input(shape=(1,))
cm = layers.concatenate([x, masks], axis=1)
outputs = layers.Dense(1)(cm)

model = tf.keras.Model(inputs=inputs + [masks], outputs=[outputs])
optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mae')
model.build([(None, 1), (None, 1), (None, 1)])             # Build has to be after compile
print('model', model.summary())

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

m_np = m_np[..., np.newaxis]
x_np = x_np[..., np.newaxis]
x_np2 = np.copy(x_np)
y_np = y_np[..., np.newaxis]

print('before', np.concatenate([x_np, y_np], axis=1))

batch_size = 1
print('dim', x_np.shape, x_np2.shape, m_np.shape)
model.fit(x=[x_np, x_np2, m_np], y=[y_np], epochs=10, batch_size=batch_size)
print('dim', x_np.shape, x_np2.shape, m_np.shape)
y_p = model.predict(x=[x_np, x_np2, m_np], batch_size=batch_size)
print('model output', model.outputs, model.inputs)

print('after', np.concatenate([x_np, y_np, y_p], axis=1))
print('model', model.summary())


cprint('test', 'blue')
