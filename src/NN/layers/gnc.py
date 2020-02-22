"""
All the needed layers for a Graph Convolutional Network
"""
import tensorflow as tf
from enum import Enum

from .KerasLayer import KerasLayer
from src import GlobalVariables as g

layers = tf.keras.layers
K = tf.keras.backend
math = tf.math


class NodeType(Enum):

    GLOBAL = 0
    LEAF = 1
    OR = 2
    AND = 3
    NOT = 4


class GraphConv(KerasLayer):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, out_features, *args, bias=True, indep_weights=True, **kwargs):
        """

        :param out_features:
        :param bias:
        :param indep_weights:
        """
        super(GraphConv, self).__init__(*args, **kwargs)
        # Raw params
        self.out_features = out_features
        self.bias = bias
        self.indep_weights = indep_weights

        # Variables
        self.weights = None
        self.weights_global = None
        self.weights_leaf = None
        self.weights_or = None
        self.weights_and = None
        self.weights_not = None
        self.weights_bias = self.add_variable(name='bias', shape=[out_features]) if bias else None

    def get_config(self):
        config = super(GraphConv, self).get_config()
        config.update(dict(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
            indep_weights=self.indep_weights
        ))
        return config

    def build(self, input_shape):
        """

        :param input_shape: [(None, (None?), in_features), adj, labels]
        """
        x_shape, adj_shape, labels_shape = input_shape
        in_features = x_shape[-1]
        if self.indep_weights:
            self.weights = self.add_variable(name='weights', shape=[in_features, self.out_features])
        else:
            # Create weights for Global, Leaf, OR, AND, NOT
            # Global
            self.weights_global = self.add_variable(name="global", shape=[in_features, self.out_features])
            # Leaf
            self.weights_leaf = self.add_variable(name="leaf", shape=[in_features, self.out_features])
            # Or
            self.weights_or = self.add_variable(name="or", shape=[in_features, self.out_features])
            # And
            self.weights_and = self.add_variable(name="and", shape=[in_features, self.out_features])
            # Not
            self.weights_not = self.add_variable(name="not", shape=[in_features, self.out_features])

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: (None, (None?), in_features)
        :return: (None, (None?), out_features)
        """
        return (*input_shape[:-1], self.out_features)

    def call(self, inputs):
        """

        :param inputs: [(None, (None?), in_features), adj, labels]
        :return: (None, (None?), out_features)
        """
        inputs, adj, labels = inputs
        if self.indep_weights:
            support = []
            for i in range(labels.shape[0]):
                if labels[i] is NodeType.GLOBAL:
                    support.append(math.multiply(inputs[i], self.weights_global))
                if labels[i] is NodeType.LEAF:
                    support.append(math.multiply(inputs[i], self.weights_leaf))
                if labels[i] is NodeType.OR:
                    support.append(math.multiply(inputs[i], self.weights_or))
                if labels[i] is NodeType.AND:
                    support.append(math.multiply(inputs[i], self.weights_and))
                if labels[i] is NodeType.NOT:
                    support.append(math.multiply(inputs[i], self.weights_not))
            support = tf.stack(value=support, axis=0)
        else:
            support = math.multiply(inputs, self.weights)
        outputs = math.multiply(adj, support)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


class GCN(KerasLayer):
    """

    """
    def __init__(self, sizes, *args, dropout=0, indep_weights=True, **kwargs):
        """

        :param sizes: List()[int]
        :param dropout:
        :param indep_weights:
        """
        super(GCN, self).__init__(*args, **kwargs)
        # Raw params
        self.sizes = sizes
        self.dropout = dropout,
        self.indep_weights = indep_weights

        # NN
        self.gc_list = [GraphConv(size) for size in sizes]
        self.dropout_layer = layers.Dropout(dropout=dropout)

    def get_config(self):
        config = super(GCN, self).get_config()
        config.update(dict(
            nfeat=self.nfeat,
            nhid=self.nhid,
            nclass=self.nclass,
            dropout=self.dropout,
            indep_weights=self.indep_weights
        ))

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        x_shape, adj_shape, labels_shape = input_shape
        for index, gc in enumerate(self.gc_list):
            gc.build([x_shape, adj_shape, labels_shape])
            x_shape = gc.compute_output_shape([x_shape, adj_shape, labels_shape])
            if index == 0:
                self.dropout_layer.build(x_shape)

    def compute_output_shape(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        x_shape, adj_shape, labels_shape = input_shape
        for gc in self.gc_list:
            x_shape = gc.compute_output_shape([x_shape, adj_shape, labels_shape])
        return x_shape

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        x, adj, labels = inputs
        for index, gc in enumerate(self.gc_list):
            x = gc([x, adj, labels])
            if index < len(self.gc_list) - 1:
                x = tf.nn.relu(x)
            if index == 0:
                x = self.dropout_layer(x)
        return x













