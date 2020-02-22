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
    def __init__(self, in_features, out_features, *args, bias=True, indep_weights=True, **kwargs):
        """

        :param in_features:
        :param out_features:
        :param bias:
        :param indep_weights:
        """
        super(GraphConv, self).__init__(*args, **kwargs)
        # Raw params
        self.in_features = in_features
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
        if self.indep_weights:
            self.weights = self.add_variable(name='weights', shape=[in_features, out_features])
        else:
            # Create weights for Global, Leaf, OR, AND, NOT
            # Global
            self.weights_global = self.add_variable(name="global", shape=[in_features, out_features])
            # Leaf
            self.weights_leaf = self.add_variable(name="leaf", shape=[in_features, out_features])
            # Or
            self.weights_or = self.add_variable(name="or", shape=[in_features, out_features])
            # And
            self.weights_and = self.add_variable(name="and", shape=[in_features, out_features])
            # Not
            self.weights_not = self.add_variable(name="not", shape=[in_features, out_features])

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

        :param input_shape: (None, (None?), in_features)
        """
        pass

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: (None, (None?), in_features)
        :return: (None, (None?), out_features)
        """
        return (*input_shape[:-1], self.out_features)

    def call(self, inputs, adj, labels=None):
        """

        :param labels:
        :param adj:
        :param inputs: (None, (None?), in_features)
        :return: (None, (None?), out_features)
        """
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

