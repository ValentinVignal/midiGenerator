from ..KerasLayer import KerasLayer


class ApplySameOnList(KerasLayer):
    def __init__(self, layer, *args, **kwargs):
        """

        :param layer: a Keras layer
        :param args:
        :param kwargs:
        """
        super(ApplySameOnList, self).__init__(*args, **kwargs)
        self.layer = layer

    def build(self, input_shape):
        self.layer.build(input_shape[0])
        super(ApplySameOnList, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_shape = self.layer.compute_output_shape(input_shape[0])
        return [output_shape for _ in input_shape]

    def call(self, x):
        return [self.layer(x_) for x_ in x]


class ApplyDifferentOnList(KerasLayer):
    def __init__(self, layers, *args, **kwargs):
        """

        :param layers: a list of Keras layer
        :param args:
        :param kwargs:
        """
        super(ApplyDifferentOnList, self).__init__(*args, **kwargs)
        self.layers = layers

    def build(self, input_shape):
        if len(input_shape) != len(self.layers):
            raise IndexError(f'Number of inputs {len(input_shape)} != number of layers {len(self.layers)}')
        for i in range(len(input_shape)):
            self.layers[i].build(input_shape[i])
        super(ApplyDifferentOnList, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return [
            self.layers[i].compute_output_shape(input_shape[i])
            for i in range(len(input_shape))
        ]

    def call(self, x):
        return [self.layers[i](x[i]) for i in range(len(self.layers))]


class ApplyDifferentLayers(KerasLayer):
    def __init__(self, layers, *args, **kwargs):
        """

        :param layers: a list of Keras layer
        :param args:
        :param kwargs:
        """
        super(ApplyDifferentLayers, self).__init__(*args, **kwargs)
        self.layers = layers

    def build(self, input_shape):
        for i in range(len(input_shape)):
            self.layers[i].build(input_shape)
        super(ApplyDifferentLayers, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return [
            self.layers[i].compute_output_shape(input_shape)
            for i in range(len(input_shape))
        ]

    def call(self, x):
        return [layer(x) for layer in self.layers]
