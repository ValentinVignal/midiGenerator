import argparse
import os
from termcolor import cprint, colored
import numpy as np
import tensorflow as tf

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

K = tf.keras.backend

from MidiGenerator.MidiGenerator import MidiGenerator
from Callbacks.LossHistory import LossHistory
import src.global_variables as g


def create_list(string):
    """

    :param string:
    :return:
    """
    string_list = string.split(':')
    if len(string_list) == 1:
        return [float(string_list[0])]
    else:
        return list(
            np.arange(float(string_list[0]), float(string_list[1]) + float(string_list[2]), float(string_list[2])))


def get_tuple(string, t=float):
    """

    :param t: float, int or other type
    :param string:
    :return:
    """
    return tuple(map(t, string.split(',')))


def create_dimensions(args):
    """

    :param args:
    :return:
    """
    dimensions = []
    default_dim = []

    # lr
    lr_tuple = get_tuple(args.lr)
    dimensions.append(Real(low=lr_tuple[0], high=lr_tuple[1], prior='log-uniform', name='lr'))
    default_dim.append((lr_tuple[0] + lr_tuple[1]) / 2)

    return dimensions, default_dim


def main(args):
    """
        Entry point
    """
    # ----- pc -----
    if args.pc:
        # args.data = 'lmd_matched_mini'
        data_path = os.path.join('../Dataset', args.data)
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'
    if args.mono:
        data_transformed_path += 'Mono'

    # Choose GPU
    if not args.pc:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dimensions, default_dim = create_dimensions(args)

    best_accuracy = 0

    def create_model(lr):
        midi_generator = MidiGenerator(name=args.name)
        midi_generator.load_data(data_transformed_path=data_transformed_path)

        opt_params = dict(
            lr=lr,
            name=args.optimizer,
            decay_drop=float(args.decay_drop),
            epoch_drop=float(args.epochs_drop)
        )
        model_options = dict(
            dropout=args.dropout,
            all_sequence=args.all_sequence,
            lstm_state=args.lstm_state,
            sampling=not args.no_sampling,
            kld=not args.no_kld
        )

        midi_generator.new_nn_model(
            model_id=args.model_id,
            opt_param=opt_params,
            work_on=args.work_on,
            type_loss=args.type_loss,
            model_options=model_options
        )
        return midi_generator

    @use_named_args(dimensions=dimensions)
    def fitness(lr):
        print(f'lr {lr:.1e}')
        midi_generator = create_model(
            lr=lr
        )
        history = midi_generator.train(epochs=args.epochs, batch=args.batch, callbacks=[], verbose=1,
                                       validation=args.validation)
        accuracy = history['val_acc'][-1]
        print(f'Accuracy - {accuracy:.2%}')

        global best_accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        midi_generator.keras_nn.clear_session()
        del midi_generator

        return -accuracy

    cprint('---------- Done ----------', 'grey', 'on_green')


def preprocess_args(args):
    """

    :param args:
    :return:
    """
    if args.pc:
        args.epochs = 2 if args.epochs == g.epochs else args.epochs
        args.seed = 2
    return args


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    parser = argparse.ArgumentParser(description='Program to train a model over a Midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_small',
                        help='The name of the data')
    # ----------------
    parser.add_argument('-e', '--epochs', type=int, default=g.epochs,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=8,
                        help='The number of the batches')
    # ----------------
    parser.add_argument('--lr', type=str, default='2:4:1',
                        help='learning rate = 10^-lr')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='Name of the optimizer (separeted with ,)(ex : adam,sgd)')
    parser.add_argument('--epochs-drop', type=str, default='50:100:50',
                        help='how long before a complete drop (decay)')
    parser.add_argument('--decay-drop', type=str, default='0.25:0.5:0.25',
                        help='0 < decay_drop < 1, every epochs_drop, lr will be multiply by decay_drop')
    parser.add_argument('--decay', type=str, default='0.25:0.5:0.25',
                        help='the value of the decay')
    parser.add_argument('--dropout', type=str, default='0.1:0.2:0.1',
                        help='Value of the dropout')
    parser.add_argument('--type-loss', type=str, default=g.type_loss,
                        help='Value of the dropout')
    parser.add_argument('--all-sequence', type=str, default='False',
                        help='Use or not all the sequence in the RNN layer (separated with ,)')
    parser.add_argument('--lstm-state', type=str, default='False',
                        help='Use or not all the sequence in the RNN layer (separated with ,)')
    # ---------------- Training options ----------------
    parser.add_argument('--noise', type=float, default=g.noise,
                        help='If not 0, add noise to the input for training')
    # ----------------
    parser.add_argument('-n', '--name', type=str, default='name',
                        help='Name given to the model')
    parser.add_argument('--work-on', type=str, default=g.work_on,
                        help='note, beat or measure')
    parser.add_argument('--mono', default=False, action='store_true',
                        help='To work with monophonic instruments')
    # ----------------
    parser.add_argument('--model-name', type=str, default='rnn',
                        help='The model name')
    parser.add_argument('--model-param', type=str, default='pc,0,1',
                        help='the model param (json file)')
    parser.add_argument('--nb-steps', type=str, default='8,16',
                        help='Nb step to train on')
    # ---------- Generation ----------
    parser.add_argument('--compare-generation', default=False, action='store_true',
                        help='Compare generation after training')
    parser.add_argument('--generate', default=False, action='store_true',
                        help='Generation after training')
    parser.add_argument('--seed', default=4,
                        help='number of seeds or the path to the folder with the seeds')
    parser.add_argument('--length', type=int, default=20,
                        help='The length of the generated music')
    parser.add_argument('--no-duration', action='store_true', default=False,
                        help='Generate only shortest notes possible')
    parser.add_argument('--verbose_generation', type=int, default=1,
                        help='Level of verbose')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Fraction of the training data to be used as validation data')
    # ---------------- Hardware options ----------------
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='To work on a small computer with a cpu')

    args = parser.parse_args()

    args = preprocess_args(args)

    main(args)
