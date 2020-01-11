import argparse
import os
from termcolor import cprint, colored
import numpy as np
import tensorflow as tf
import random

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

K = tf.keras.backend

from src.MidiGenerator import MidiGenerator
from src.NN.Callbacks import LossHistory
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


def get_tuple(string, t=float, separator=':'):
    """
    Get the tuple from the string given by the user
    :param separator: separator of the value in the string
    :param t: float, int or other type (function)
    :param string: the string given by the user
    :return:
    """
    return tuple(map(t, string.split(separator)))


def str2bool(string):
    return string == 'True'


def create_dimensions(args):
    """

    :param args:
    :return:
    """
    dimensions = []
    default_dim = []

    def add_Real(mtuple, name, prior='uniform'):
        if len(mtuple) == 1:
            dimensions.append(Categorical([mtuple[0]], name=name))
            default_dim.append(mtuple[0])
        elif len(mtuple) == 2:
            dimensions.append(Real(low=min(mtuple), high=max(mtuple), name=name, prior=prior))
            default_dim.append(sum(mtuple) / len(mtuple))

    def add_Categorical(m_tuple, name):
        dimensions.append(Categorical(list(m_tuple), name=name))
        default_dim.append(m_tuple[0])

    # lr
    lr_tuple = get_tuple(args.lr)
    lr_tuple = tuple(map(lambda x: 10**(-x), lr_tuple))
    add_Real(lr_tuple, name='lr', prior='log-uniform')
    # optimizer
    opt_tuple = get_tuple(args.optimizer, t=str, separator=',')
    add_Categorical(opt_tuple, name='optimizer')
    # decay
    decay_tuple = get_tuple(args.decay)
    add_Real(decay_tuple, name='decay')
    # dropout
    dropout_tuple = get_tuple(args.dropout)
    add_Real(dropout_tuple, name='dropout')
    # sampling
    sampling_tuple = get_tuple(args.no_sampling, t=lambda x: not str2bool(x), separator=',')
    add_Categorical(sampling_tuple, name='sampling')
    # kld
    kld_tuple = get_tuple(args.no_kld, t=lambda x: not str2bool(x), separator=',')
    add_Categorical(kld_tuple, name='kld')
    # all sequence
    all_sequence_tuple = get_tuple(args.all_sequence, t=str2bool, separator=',')
    add_Categorical(all_sequence_tuple, name='all_sequence')
    # model name
    model_name_tuple = get_tuple(args.model_name, t=str, separator=',')
    add_Categorical(model_name_tuple)
    # model param
    model_param_tuple = get_tuple(args.model_param, t=str, separator=',')
    add_Categorical(model_param_tuple)
    # nb steps
    nb_steps_tuple = get_tuple(args.nb_steps, t=int, separator=',')
    add_Categorical(nb_steps_tuple)

    return dimensions, default_dim


def get_history_acc(history):
    accuracy = 0
    i = 0
    while f'val_Output_{i}_acc' in history:
        accuracy += history[f'val_Output_{i}_acc'][-1]
        i += 1
    return accuracy / i


def str_hp_to_print(name, value, exp_format=False, first_printed=False):
    """

    :param name:
    :param value:
    :param exp_format:
    :param first_printed:
    :return: string which is pretty to print (to show current hp tested)
    """
    s = ''
    if not first_printed:
        s += ' - '
    s += f'{name}: '
    if exp_format:
        s += colored(f'{value:.1e}', 'magenta')
    else:
        s += colored(f'{value}', 'magenta')
    return s

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------


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

    global best_accuracy
    best_accuracy = 0

    def create_model(lr, optimizer, decay, dropout, sampling, kld, all_sequence, model_name, model_param, nb_steps):
        midi_generator = MidiGenerator(name=args.name)
        midi_generator.load_data(data_transformed_path=data_transformed_path, verbose=0)

        model_id = f'{model_name},{model_param},{nb_steps}'

        opt_params = dict(
            lr=lr,
            name=optimizer,
            decay_drop=float(args.decay_drop),
            epoch_drop=float(args.epochs_drop),
            decay=decay
        )
        model_options = dict(
            dropout=dropout,
            all_sequence=all_sequence,
            lstm_state=args.lstm_state,
            sampling=sampling,
            kld=kld
        )

        midi_generator.new_nn_model(
            model_id=model_id,
            opt_param=opt_params,
            work_on=args.work_on,
            type_loss=args.type_loss,
            model_options=model_options,
            print_model=False
        )
        return midi_generator

    @use_named_args(dimensions=dimensions)
    def fitness(lr, optimizer, decay, dropout, sampling, kld, all_sequence, model_name, model_param, nb_steps):
        s = ''
        model_id = f'{model_name},{model_param},{nb_steps}'
        s += str_hp_to_print('model', model_id, first_printed=True)
        s += str_hp_to_print('lr', lr, exp_format=True)
        s += str_hp_to_print('opt', optimizer)
        s += str_hp_to_print('decay', decay, exp_format=True)
        s += str_hp_to_print('dropout', dropout, exp_format=True)
        s += str_hp_to_print('sampling', sampling)
        s += str2bool('kld', kld)
        s += str_hp_to_print('all_sequence', all_sequence)

        print(s)

        midi_generator = create_model(
            lr=lr,
            optimizer=optimizer,
            decay=decay,
            dropout=dropout,
            sampling=sampling,
            kld=kld,
            all_sequence=all_sequence,
            model_name=model_name,
            model_param=model_param,
            nb_steps=nb_steps
        )
        history = midi_generator.train(epochs=args.epochs, batch=args.batch, callbacks=[], verbose=1,
                                       validation=args.validation)
        accuracy = get_history_acc(history)
        print(f'Accuracy - {accuracy:.2%}')

        global best_accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        midi_generator.keras_nn.clear_session()
        del midi_generator

        return -accuracy

    search_result = gp_minimize(
        func=fitness,
        dimensions=dimensions,
        acq_func='EI',
        n_calls=20,
        x0=default_dim
    )

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
    parser.add_argument('--lr', type=str, default='2:4',
                        help='learning rate = 10^-lr')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='Name of the optimizer (separeted with ,)(ex : adam,sgd)')
    parser.add_argument('--epochs-drop', type=int, default=50,#'50:100:50',
                        help='how long before a complete drop (decay)')
    parser.add_argument('--decay-drop', type=float, default=0.25,#'0.25:0.5:0.25',
                        help='0 < decay_drop < 1, every epochs_drop, lr will be multiply by decay_drop')
    parser.add_argument('--decay', type=str, default='0.01:1',
                        help='the value of the decay')
    parser.add_argument('--dropout', type=str, default='0.1:0.2',
                        help='Value of the dropout')
    parser.add_argument('--type-loss', type=str, default=g.type_loss,
                        help='Value of the dropout')
    parser.add_argument('--all-sequence', type=str, default='False',
                        help='Use or not all the sequence in the RNN layer (separated with ,)')
    parser.add_argument('--lstm-state', type=bool, default=False,#'False',
                        help='Use or not all the sequence in the RNN layer (separated with ,)')
    parser.add_argument('--no-sampling', type=str, default='False',
                        help='Gaussian Sampling')
    parser.add_argument('--no-kld', default=False, action='store_true',
                        help='No KL Divergence')
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
    parser.add_argument('--nb-steps', type=str, default='4',#'8,16',
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
