import tensorflow as tf
import warnings
import numpy as np

from .KerasLayer import KerasLayer
from src import mtypes as t


def get_shape(t, ax):
    """

    :param t:
    :param ax:
    :return:
    """
    if t[ax] is None or isinstance(t[ax], int):
        return t[ax]
    else:
        return t[ax].value


class SwitchListAxis(KerasLayer):
    """

    """

    def __init__(self, axis=0, *args, **kwargs):
        """

        :param axis: Batch axis is not included
        """
        super(SwitchListAxis, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.axis =axis

        self.perms = None

    def get_config(self):
        config = super(SwitchListAxis, self).get_config()
        config.update(dict(
            axis=self.axis
        ))
        return config

    def build(self, input_shape):
        self.verify_shapes(input_shape)
        self.get_perms(input_shape)
        super(SwitchListAxis, self).build(input_shape)

    def get_perms(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        nb_dim = len(input_shape[0]) + 1        # + new axis from stack
        perms = list(range(nb_dim))
        perms[0] = self.axis + 2        # batch + new axis from stack
        perms[self.axis+2] = 0
        self.perms = perms

    @staticmethod
    def verify_shapes(input_shape):
        for i in range(1, len(input_shape)):
            if not is_equal_tuple(input_shape[i - 1], input_shape[i], warn=False):
                warnings.warn(
                    f"All the shapes should be have the same dimension: shape {i - 1}: {input_shape[i - 1]} != {i}: {input_shape[i]}"
                )

    def compute_output_shape(self, input_shape):
        len_list = len(input_shape)
        axis_with_batch = self.axis + 1
        new_len_list = input_shape[0][axis_with_batch]

        return [(*input_shape[0][:axis_with_batch], len_list, *input_shape[0][axis_with_batch+1:]) for i in range(new_len_list)]

    def call(self, inputs):
        """

        :param inputs: List[tensor]
        :return:
        """
        inputs = tf.stack(inputs, axis=0)       # Tensor
        outputs = tf.transpose(inputs, perm=self.perms)     # Tensor
        outputs = tf.unstack(outputs)       # List[tensor]
        return outputs


class Stack(KerasLayer):
    def __init__(self, axis: t.Any = 0, axis_reversed=False, *args, **kwargs):
        """

        :param axis: No batch in axis
        """
        super(Stack, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.param_axis = axis
        self.param_axis_reversed = axis_reversed

        self.axis = None
        self.axis_with_batch = None
        self.compute_axis(axis)
        self.axis = self.axis[::-1] if axis_reversed else self.axis

    def compute_axis(self, axis):
        if isinstance(axis, int):
            axis = [axis]
        elif isinstance(axis, tuple):
            axis = list(axis)
        assert isinstance(axis, list)
        self.axis = axis

    def get_config(self):
        config = super(Stack, self).get_config()
        config.update(dict(
            axis=self.param_axis,
            axis_reversed=self.param_axis_reversed
        ))
        return config

    def build(self, input_shape):
        for ax in self.axis:
            input_shape = input_shape[0]
        self.compute_axis_with_batch(input_shape)
        super(Stack, self).build(input_shape)

    def compute_axis_with_batch(self, input_shape):
        axis_with_batch = []
        nb_dims = len(input_shape)      # (with batch in it)
        for ax in self.axis:
            if ax >= 0:
                real_ax = ax + 1
            else:
                real_ax = nb_dims + 1 + ax
            for ax_ in axis_with_batch:
                if ax_ <= real_ax:
                    real_ax -= 1
            axis_with_batch.append(real_ax)
        self.axis_with_batch = axis_with_batch

    def compute_output_shape(self, input_shape):
        all_length = []
        for ax in self.axis_with_batch:
            all_length.append(len(input_shape))
            input_shape = input_shape[0]
        output_shape = list(input_shape)
        for i in range(len(self.axis_with_batch) - 1, -1, -1):
            output_shape.insert(self.axis_with_batch[i], all_length[i])
        return tuple(output_shape)

    def recurse_stack(self, x, axis):
        if isinstance(x, list):
            x = [self.recurse_stack(x=x_, axis=axis[1:]) for x_ in x]
            return tf.stack(x, axis=axis[0])
        else:
            return x

    def call(self, x):
        x = self.recurse_stack(x, self.axis_with_batch)
        return x


def is_equal_tuple(t1, t2, warn=False):
    """

    :param t1:
    :param t2:
    :param warn:
    :return:
    """
    if len(t1) != len(t2):
        if warn:
            warnings.warn(f'Tuples don t have same length : len({t1}) = {len(t1)} != len({t2}) = {len(t2)}')
        return False
    else:
        for v1, v2 in zip(t1, t2):
            if v1 != v2:
                if warn:
                    warnings.warn(f'Tuples are not equal : {t1} != {t2}')
                return False
        return True


class Unstack(KerasLayer):

    def __init__(self, axis: t.Any = 0, axis_reversed=False, *args, **kwargs):
        """

        :param axis:
        :param axis_reversed:
        """
        super(Unstack, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.param_axis = axis
        self.param_axis_reversed = axis_reversed

        self.axis = None,
        self.axis_with_batch = None
        self.compute_axis(axis)
        self.axis = self.axis[::-1] if axis_reversed else self.axis

    def compute_axis(self, axis):
        if isinstance(axis, int):
            axis = [axis]
        elif isinstance(axis, tuple):
            axis = list(axis)
        assert isinstance(axis, list)
        self.axis = axis

    def get_config(self):
        config = super(Unstack, self).get_config()
        config.update(dict(
            axis=self.param_axis,
            axis_reversed=self.param_axis_reversed
        ))
        return config

    def build(self, input_shape):
        self.compute_axis_with_batch(input_shape)
        super(Unstack, self).build(input_shape)

    def compute_axis_with_batch(self, input_shape):
        axis_with_batch = []
        nb_dims = len(input_shape)      # (with batch in it)
        for ax in self.axis:
            if ax >= 0:
                real_ax = ax + 1
            else:
                real_ax = nb_dims + 1 + ax
            for ax_ in axis_with_batch:
                if ax_ <= real_ax:
                    real_ax -= 1
            axis_with_batch.append(real_ax)
        self.axis_with_batch = axis_with_batch

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        lengths = []
        for ax in reversed(self.axis_with_batch):
            lengths.append(input_shape.pop(ax))
        output_shape = self.recurse_compute_output_shape(input_shape, lengths)
        return output_shape

    def recurse_compute_output_shape(self, input_shape, lengths):
        if not lengths:
            return input_shape
        else:
            return [
                self.recurse_compute_output_shape(input_shape, lengths=lengths[1:]) for i in range(lengths[0])
            ]

    def call(self, x):
        return self.recurse_call(x, self.axis_with_batch)

    def recurse_call(self, x, axis):
        if not axis:
            return x
        else:
            x = tf.unstack(x, axis=axis[0])
            x = [self.recurse_call(x_, axis[1:]) for x_ in x]
            return x


class ExpandDims(KerasLayer):
    def __init__(self, axis=0, *args, **kwargs):
        super(ExpandDims, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.axis = axis

        self.axis_with_batch = axis + 1 if axis >= 0 else axis

    def get_config(self):
        config = super(ExpandDims, self).get_config()
        config.update(dict(
            axis=self.axis
        ))
        return config

    def compute_output_shape(self, input_shape):
        return (*input_shape[:self.axis_with_batch], 1, *input_shape[self.axis_with_batch:])

    def call(self, inputs):
        return tf.expand_dims(inputs, self.axis_with_batch)


def transpose_list(l, axes):
    """

    :param l: list of object (not np array)
    :param axes:
    :return:
    """
    l_np = np.asarray(l)
    l_np = np.transpose(l_np, axes=axes)

    def recurse_np2list(array, nb_lists):
        """

        :param array:
        :param nb_lists:
        :return:
        """
        if nb_lists == 0:
            return array
        else:
            return [
                recurse_np2list(arr, nb_lists - 1) for arr in array
            ]
    return recurse_np2list(l_np, len(axes))






