import tensorflow as tf
import numpy as np
import argparse
import json
import sys

if __name__ == '__main__':
    sys.path.append(sys.path[0] + '\\..\\..\\..\\..')

import src.NN.losses as mlosses
import src.NN.metrics as mmetrics
import src.global_variables as g
import src.NN.layers as mlayers
from src.NN.Models.KerasModel import KerasModel
import src.NN.shapes.convolution as s_conv
import src.NN.shapes.time as s_time
from src.eval_string import eval_object

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda
K = tf.keras.backend


def create_model(input_param, model_param, nb_steps, step_length, optimizer, type_loss=g.type_loss,
                 model_options={}
                 ):
    """

    :param input_param: {
                            nb_instruments,
                            input_size
                        }
                        (Comes from the dataset)
    :param model_param: (Comes from the .json)
    :param nb_steps: (Comes from the user)
    :param step_length: (Comes from the user)
    :param optimizer:
    :param type_loss: (Comes from the user)
    :param model_options: (Comes from the user)
    :return: the neural network:
    """

    # ---------- Model options ----------
    mmodel_options = {
        'dropout': g.dropout,
        'lambdas_loss': g.lambdas_loss,
        'sample': g.sample
    }
    mmodel_options.update(model_options)

    dropout = mmodel_options['dropout']

    lambda_loss_activation, lambda_loss_duration = g.get_lambdas_loss(mmodel_options['lambdas_loss'])

    print('Definition of the graph ...')

    # --------- End model options ----------

    # --------- Variables ----------

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    model_param = eval_object(model_param, env=env)

    time_stride = s_time.time_stride(step_length)

    midi_shape = (nb_steps, step_length, input_size, 1)  # (batch, step_length, nb_step, input_size, 2)
    mask_shape = (nb_instruments, nb_steps)

    # ----- For the decoder -----

    model_param_dec, shape_before_conv_dec = s_conv.compute_model_param_dec(
        input_shape=(*midi_shape[:-1], nb_instruments),
        model_param_enc=model_param,
        strides=[(1, time_stride, 2) for i in range(len(model_param['conv']) - 1)]
    )

    shapes_before_pooling = s_conv.compute_shapes_before_pooling(
        input_shape=midi_shape,
        model_param_conv=model_param['conv'],
        strides=(1, time_stride, 2)
    )
    # Put time to 1 :
    shapes_before_pooling = s_time.time_step_to_x(l=shapes_before_pooling, axis=1, x=1)

    # --------- End Variables ----------

    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size, 2)]
    input_mask = tf.keras.Input(mask_shape)  # (batch, nb_instruments, nb_steps)

    # ---------- All together ----------
    inputs_encoded = [mlayers.coder3D.Encoder3D(
        encoder_param=model_param,
        dropout=dropout,
        time_stride=time_stride
    )(input_midi) for input_midi in inputs_midi]  # List(nb_instruments)[(batch, nb_steps, size)]

    latent_size = model_param['dense'][-1]
    means = [mlayers.dense.DenseForMean(units=latent_size)(x) for x in
             inputs_encoded]  # List(nb_instruments)[(batch, nb_steps, size)]
    stds = [mlayers.dense.DenseForSTD(units=latent_size)(x) for x in
            inputs_encoded]  # List(nb_instruments)[(batch, nb_steps, size)]
    means = mlayers.shapes.Stack(axis=0)(means)  # (batch, nb_instruments, nb_steps, size)
    stds = mlayers.shapes.Stack(axis=0)(stds)  # (batch, nb_instruments, nb_steps, size)

    poe = mlayers.vae.ProductOfExpertMask(axis=0)([means, stds, input_mask])  # List(2)[(batch, nb_steps, size)]
    kld = mlayers.vae.KLD()(poe)
    if mmodel_options['sample']:
        samples = mlayers.vae.SampleGaussian()(poe)  # (batch, nb_steps, size)
    else:
        samples = layers.concatenate(poe)

    x = mlayers.rnn.LstmRNN(
        size_list=model_param['lstm'],
        return_sequence=True)(samples)
    x = mlayers.coder3D.Decoder3D(decoder_param=model_param_dec,
                                  shape_before_conv=shape_before_conv_dec,
                                  dropout=dropout,
                                  time_stride=time_stride,
                                  shapes_after_upsize=shapes_before_pooling)(x)
    outputs = mlayers.last.LastMono(softmax_axis=-2)(
        x)  # List(nb_instruments)[(batch, nb_steps=1, step_size, input_size, 1)]
    outputs = [layers.Layer(name=f'Output_{inst}')(outputs[inst]) for inst in range(nb_instruments)]

    model = KerasModel(inputs=inputs_midi + [input_mask], outputs=outputs)

    # ------------------ Losses -----------------
    # Define losses dict for outputs
    losses = {}
    for inst in range(nb_instruments):
        losses[f'Output_{inst}'] = mlosses.loss_function_mono

    # Define kld
    model.add_loss(kld)

    # ------------------ Metrics -----------------

    # ------------------------------ Compile ------------------------------

    model.compile(loss=losses,
                  optimizer=optimizer,
                  metrics=[mmetrics.acc_mono])
    model.build([(None, *midi_shape) for inst in range(nb_instruments)])

    return model, losses, (lambda_loss_activation, lambda_loss_duration)


def create_fake_data(input_shape, size=20):
    data_x = [np.zeros((size, *input_shape[1:])) for i in range(input_shape[0])]
    output_shape = (input_shape[0], *input_shape[2:])
    data_y = [np.ones((size, *output_shape[1:])) for i in range(output_shape[0])]
    print('data y', np.asarray(data_y).shape)

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
    parser = argparse.ArgumentParser(description='Program to train a model over a Midi dataset',
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
