import tensorflow as tf

from .KerasLayer import KerasLayer

math = tf.math
layers = tf.keras.layers


class TutoSA(KerasLayer):
    """
    Self attention of the tutorial
    """
    def __init__(self, *args, **kwargs):
        super(TutoSA, self).__init__(*args, **kwargs)
        self.dense = layers.dense(1, activation='tanh')(x)
        self.flatten = layers.Flatten()
        self.activation = layers.Softmax()
        self.repeat_vector = None
        self.permute = layers.Permute([2, 1])
        self.multiply = layers.Multiply()
        self.dense2 = None

    def get_config(self):
        return super(TutoSA, self).get_config()

    def build(self, input_shape):
        """

        :param input_shape: (batch, nb_steps, input_size)
        :return:
        """
        super(TutoSA, self).build(input_shape)
        input_size = input_shape[-1]
        self.dense.build(input_shape)
        attention_shape = self.dense.compute_output_shape(input_shape)
        self.flatten.build(attention_shape)
        attention_shape = self.flatten.compute_output_shape(attention_shape)
        self.activation.build(attention_shape)
        self.repeat_vector = layers.RepeatVector(input_size)
        self.repeat_vector.build(attention_shape)
        attention_shape = self.repeat_vector.compute_output_shape(attention_shape)
        self.multiply.build([attention_shape, input_shape])
        attention_shape = self.multiply.compute_output_shape([attention_shape, input_shape])
        self.dense2 = layers.Dense(input_size)
        self.dense2.build(attention_shape)

    def call(self, inputs):
        attention = self.dense(inputs)
        attention = self.flatten(attention)
        attention = self.activation(attention)
        attention = self.repeat_vector(attention)
        attention = self.permute(attention)
        attention = self.multiply(attention)
        attention = self.dense2(attention)
        return attention

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaleDotProductAttention(KerasLayer):

    def __init__(self, latent_size=None, *args, **kwargs):
        super(ScaleDotProductAttention, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.latent_size = latent_size

        self.dense_q = layers.Dense(latent_size)
        self.dense_k = layers.Dense(latent_size)
        self.dense_v = layers.Dense(latent_size)

    def from_config(self):
        config = super(ScaleDotProductAttention, self).from_config()
        config.update(dict(
            latent_size=self.latent_size
        ))
        return config

    def build(self, input_shape):
        super(ScaleDotProductAttention, self).build(input_shape)
        if self.latent_size is None:
            self.latent_size = input_shape[-1]
        self.dense_k.build(input_shape)
        self.dense_v.build(input_shape)
        self.dense_q.build(input_shape)

    def call(self, inputs):
        """

        :param inputs: (batch, nb_steps, input_size)
        :return:
        """
        q = self.dense_q(inputs)
        k = self.dense_k(inputs)
        v = self.dense_v(inputs)        # (batch, nb_steps, latent_size)

        m = math.reduce_sum(math.multiply(q, k), axis=2)       # (batch, nb_steps)
        s = tf.nn.softmax(m, axis=1)        # (batch, nb_steps)

        res = v * tf.expand_dims(s, 2)      # (batch, nb_steps, latent_size)
        return res

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: (batch, nb_steps, input_size)
        :return:
        """
        return (*input_shape[:-1], self.latent_size)


class SAH(KerasLayer):
    """

    """
    def __init__(self, latent_size=None, nb_matrices=1, *args, **kwargs):
        super(SAH, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.latent_size = latent_size
        self.nb_matrices = nb_matrices

        self.sdpas = [ScaleDotProductAttention(latent_size=latent_size) for _ in range(nb_matrices)]
        self.dense = layers.Dense(latent_size)

    def from_config(self):
        config = super(SAH, self).from_config()
        config.update(dict(
            latent_size=self.latent_size,
            nb_matrixes=self.nb_matrices
        ))
        return config

    def build(self, input_shape):
        """

        :param input_shape: (batch, nb_steps, input_size)
        :return:
        """
        super(SAH, self).build(input_shape)
        if self.latent_size is None:
            self.latent_size = input_shape[-1]
        for sdpa in self.sdpas:
            sdpa.build(input_shape)
        input_shape_dense = (*input_shape[:-1], self.latent_size * self.nb_matrices)
        self.dense.build(input_shape_dense)

    def call(self, inputs):
        sdpas = [sdpa(inputs) for sdpa in self.sdpas]
        concat = tf.concat(sdpas, axis=-1)
        output = self.dense(concat)
        return output

    def compute_output_shape(self, input_shape):
        """

        :param input_shape:  (batch, nb_steps, input_size)
        :return:
        """
        return (*input_shape[:-1], self.latent_size)




