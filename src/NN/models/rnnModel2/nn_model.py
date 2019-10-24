import tensorflow as tf
import warnings
import copy
import numpy as np
import argparse
import json
import sys

if __name__ == '__main__':
    sys.path.append(sys.path[0] + '\\..\\..\\..\\..')

import src.eval_string as es
import src.NN.losses as l
import src.global_variables as g
import src.NN.layers as mlayers
import src.mtypes as t
from src.NN.models.KerasModel import KerasModel

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda
K = tf.keras.backend

"""

LSTM with encoder convolutional layers and the deconder use transposed convolutional layer

"""


def create_model(input_param, model_param, nb_steps, step_length, optimizer, type_loss=g.type_loss,
                 model_options={}
                 ):
    """

    :param input_param:
    :param nb_steps:
    :param model_param:
    :param step_length:
    :param optimizer:
    :param type_loss:
    :param model_options:
    :return: the neural network:
    """

    # ---------- Model options ----------
    mmodel_options = {
        'dropout': g.dropout,
        'lambdas_loss': g.lambdas_loss,
    }
    mmodel_options.update(model_options)

    dropout = mmodel_options['dropout']

    lambda_loss_activation, lambda_loss_duration = g.get_lambdas_loss(mmodel_options['lambdas_loss'])

    print('Definition of the graph ...')

    # --------- End model options ----------

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    if step_length == 4 or step_length == 1:
        # It is working either on 'Beat' or on 'Note'
        # We don't want to stride on the time axis
        time_stride = 1
    elif step_length == 16:
        # It is working on 'Measure'
        # We want to stride on the time axis (the size is considered as big enough)
        time_stride = 2
    else:
        warnings.warn(
            f'The model is not designed wo work with a step_length {step_length} not included in (8, 16),' +
            'some errors might occur',
            Warning)
        time_stride = 1 if step_length < 16 else 2

    midi_shape = (nb_steps, step_length, input_size, 1)  # (batch, step_length, nb_step, input_size)

    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    class MyModel(KerasModel):

        type_model_param_conv = t.List[t.List[int]]
        type_model_param = t.Dict[str, t.Union[
            type_model_param_conv,  # conv
            t.List[int]  # dense, lstm
        ]]

        def __init__(self, input_shape: t.shape, model_param: type_model_param, **kwargs):
            """

            :param input_shape:
            :param model_param:
            :param kwargs:
            """
            super(MyModel, self).__init__(name='MyModel', **kwargs)
            self.model_param_enc = model_param
            self.model_param_dec, shape_before_conv_dec = MyModel.compute_model_param_dec(input_shape,
                                                                                          self.model_param_enc)
            self.shapes_before_pooling = MyModel.compute_shapes_before_pooling(
                input_shape=input_shape,
                model_param_conv=self.model_param_enc['conv']
            )
            self.encoder = mlayers.coder3D.Encoder3D(encoder_param=self.model_param_enc,
                                                     dropout=dropout,
                                                     time_stride=time_stride)
            self.rnn = mlayers.rnn.LstmRNN(
                model_param['lstm'])  # TODO : Put eval in it and declare nb_instrument blablabla...
            self.decoder = mlayers.coder3D.Decoder3D(decoder_param=self.model_param_dec,
                                                     shape_before_conv=shape_before_conv_dec,
                                                     dropout=dropout,
                                                     time_stride=time_stride,
                                                     shapes_after_upsize=self.shapes_before_pooling)
            self.last_layer = mlayers.last.LastMono(softmax_axis=2)

        @staticmethod
        def compute_shapes_before_pooling(input_shape: t.shape, model_param_conv: type_model_param_conv) -> t.List[
            t.bshape]:
            """

            :param input_shape:
            :param model_param_conv:
            :return:
            """
            # Create the fake filters : model_param_conv = [[a, b], [c], [d, e]]
            fake_model_param_conv_temp = copy.deepcopy(model_param_conv)
            # -> [[nb_instruments, a, b], [b, c], [c, d, e]] -> [a, b, d]       (shape before pooling)
            fake_model_param_conv_temp[0].insert(0, nb_instruments)
            for i in range(1, len(model_param_conv)):
                fake_model_param_conv_temp[i].insert(0, fake_model_param_conv_temp[i - 1][-1])
            fake_model_param_conv: t.List[int] = [l[-2] for l in fake_model_param_conv_temp]

            new_shapes: t.List[t.shape] = mlayers.conv.new_shapes_conv(
                input_shape=(1, *input_shape[1:-1], fake_model_param_conv[0]),
                strides_list=[(1, time_stride, 2) for i in range(len(model_param_conv) - 1)],
                filters_list=fake_model_param_conv[1:]
            )
            final_shapes = new_shapes[::-1]
            final_bshapes = [t.Bshape.cast_from(shape, t.shape) for shape in final_shapes]
            return final_bshapes

        @staticmethod
        def compute_model_param_dec(input_shape: t.shape,
                                    model_param_enc: type_model_param
                                    ) -> t.Tuple[type_model_param, t.shape]:
            conv_enc = model_param_enc['conv']
            dense_enc = model_param['dense']

            # --- Compute the last size of the convolution (so we can add a dense layer of this size in the decoder ---
            nb_pool = len(conv_enc) - 1  # nb_times there is a stride == 2 in the conv encoder
            # 1. compute the last shape
            last_shapes_conv_enc = mlayers.conv.new_shapes_conv(input_shape=(*input_shape[:-1], conv_enc[-1][-1]),
                                                                strides_list=[(1, time_stride, 2) for i in
                                                                              range(nb_pool)],
                                                                filters_list=[conv_enc[-1][-1] for i in range(nb_pool)])
            last_shape_conv_enc = last_shapes_conv_enc[-1]
            # 2. compute the last size
            last_size_conv_enc = 1
            for i, s in enumerate(last_shape_conv_enc[1:]):  # Don't take the time axis (1 step only in decoder)
                last_size_conv_enc *= s

            # --- Create the dictionnary to return ---
            model_param_dec = dict(
                dense=dense_enc[::-1] + [last_size_conv_enc],
                conv=mlayers.conv.reverse_conv_param(original_dim=nb_instruments, param_list=conv_enc)
            )
            return model_param_dec, (1, *last_shape_conv_enc[1:])

        def build(self, input_shape):
            new_shape = (*input_shape[0][:-1], input_shape[0][-1] * len(input_shape))
            self.encoder.build(new_shape)
            new_shape = self.encoder.compute_output_shape(new_shape)
            self.rnn.build(new_shape)
            new_shape = self.rnn.compute_output_shape(new_shape)
            self.decoder.build(new_shape)
            new_shape = self.decoder.compute_output_shape(new_shape)
            self.last_layer.build(new_shape)
            self.set_weights_variables(self.encoder, self.rnn, self.decoder, self.last_layer)
            super(MyModel, self).build(input_shape)

        def call(self, inputs):
            """
            print('MyModel call inputs', inputs)
            x = [self.inputs[inst](inputs[inst]) for inst in range(nb_instruments)]
            print('MyModel call x1:', x)
            """
            x = tf.concat(inputs, axis=4)  # (batch, nb_steps, step_length, input_size, nb_instruments)
            x = self.encoder(x)
            x = self.rnn(x)
            x = self.decoder(x)
            output = self.last_layer(x)
            return output

    model = MyModel(input_shape=midi_shape, model_param=model_param)

    # ------------------ Losses -----------------
    # Define losses dict
    losses = {}
    for inst in range(nb_instruments):
        losses[f'Output_{inst}'] = l.loss_function_mono

    model.compile(loss=[tf.keras.losses.binary_crossentropy for inst in range(nb_instruments)],
                  optimizer=optimizer)  # , metrics=[l.acc_mono])
    model.build([(None, *midi_shape) for inst in range(nb_instruments)])

    return model, losses, (lambda_loss_activation, lambda_loss_duration)


def create_fake_data(input_shape, size=20):
    data_x = [np.zeros((size, *input_shape[1:])) for i in range(input_shape[0])]
    output_shape = (input_shape[0], 1, *input_shape[2:])
    data_y = [np.ones((size, *output_shape[1:])) for i in range(output_shape[0])]

    return data_x, data_y


class MS(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size=4):
        self.x = x
        self.y = y
        self.nb_instruments = len(x)
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x[0]) / self.batch_size)

    def __getitem__(self, item):
        x = [self.x[inst][item * self.batch_size: (item + 1) * self.batch_size] for inst in range(self.nb_instruments)]
        y = [self.y[inst][item * self.batch_size: (item + 1) * self.batch_size] for inst in range(self.nb_instruments)]
        return x, y


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-shape', type=str, default='4,3,4,40,1',
                        help='The input shape : nb_instruments,nb_steps,step_size,input_size,channels')
    parser.add_argument('--model-param', type=str, default='pc',
                        help='The name of the json file')
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=2,
                        help='The number of the batches')
    parser.add_argument('--lr', type=float, default=g.lr,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=g.dropout,
                        help='Value of the dropout')
    parser.add_argument('--nb-data', type=int, default=20,
                        help='Value of the dropout')
    parser.add_argument('--fit', action='store_true', default=False,
                        help='To fit the model')
    parser.add_argument('--fit-generator', action='store_true', default=False,
                        help='To fit the model with a generator')

    args = parser.parse_args()

    # ----------
    input_param = dict(nb_instruments=int(args.input_shape.split(',')[0]),
                       input_size=int(args.input_shape.split(',')[3]))
    # ----------
    with open(args.model_param + '.json') as json_file:
        model_param = json.load(json_file)
    # ----------
    nb_steps = int(args.input_shape.split(',')[1])
    # ----------
    step_length = int(args.input_shape.split(',')[2])
    # ----------
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    # ----------
    model_options = dict(dropout=args.dropout)
    # ----------
    model = create_model(input_param=input_param,
                         model_param=model_param,
                         nb_steps=nb_steps,
                         step_length=step_length,
                         optimizer=optimizer,
                         type_loss=None,
                         model_options=model_options)[0]
    print(model.summary())

    data_x, data_y = create_fake_data(tuple([int(x) for x in args.input_shape.split(',')]), size=args.nb_data)

    generator = MS(x=data_x, y=data_y, batch_size=args.batch)

    print('before arg fit')
    if args.fit:
        print('in arg fit', model.outputs)
        res = model.fit(x=data_x,
                        y=data_y,
                        batch_size=args.batch,
                        epochs=args.epochs)
    if args.fit_generator:
        print('in arg fit', model.outputs)
        res = model.fit_generator(generator=generator,
                                  epochs=args.epochs)
    print('model output names', model.output_names)
