import argparse
from enum import Enum

from src import global_variables as g


class ArgType(Enum):
    """

    """
    ALL = 0
    Train = 1
    HPSearch = 2
    Generate = 3
    Data = 10
    CheckData = 11


def get_type(argtype, t=float):
    """

    :param argtype:
    :param t:
    :return:
    """
    if argtype is ArgType.Train:
        return str
    else:
        return t


def add_store_true(parser, name, argtype=ArgType.ALL, help=''):
    """
    Add to the parser a store true action
    :param parser:
    :param name:
    :param argtype:
    :param help:
    :return:
    """
    if argtype is not ArgType.HPSearch:
        parser.add_argument(name, action='store_true',
                            help=help)
    else:
        parser.add_argument(name, type=str,
                            help=help)


def create_parser(argtype):
    """

    :param argtype:
    :return:
    """
    description = 'Default description'
    if argtype is ArgType.Train:
        description = 'To train the model'
    elif argtype is ArgType.HPSearch:
        description = 'To find the best Hyper Parameters'
    elif argtype is ArgType.Generate:
        description = 'To Generate music from a trained model'
    elif argtype is ArgType.Data:
        description = 'To compute the data'
    elif argtype is ArgType.CheckData:
        description = 'To Check the data'

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser


def add_train_args(parser, argtype=ArgType.ALL):
    """

    :param parser:
    :param argtype:
    :return:
    """
    parser.add_argument('-e', '--epochs', type=int,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch', type=int,
                        help='The number of the batches')
    parser.add_argument('--noise', type=float,
                        help='If not 0, add noise to the input for training')
    parser.add_argument('--validation', type=float,
                        help='Fraction of the training data to be used as validation data')

    # ---------- Default values ----------
    parser.set_defaults(
        epochs=g.epochs,
        batch=g.batch,
        noise=g.noise,
        validation=g.validation
    )
    return parser


def add_evaluate_model_args(parser, argtype=ArgType.ALL):
    """

    :param parser:
    :param argtype:
    :return:
    """
    parser.add_argument('--evaluate', default=False, action='store_true',
                        help='Evaluate the model after the training')
    parser.add_argument('--check-batch', type=int, default=-1,
                        help='Batch to check')
    return parser


def add_create_model_args(parser, argtype=ArgType.ALL):
    """

    :param parser:
    :param argtype:
    :return:
    """
    # -------------------- Training parameters --------------------
    parser.add_argument('--lr', type=get_type(argtype, float),
                        help='learning rate')
    parser.add_argument('-o', '--optimizer', type=str,
                        help='Name of the optimizer')
    parser.add_argument('--epochs-drop', type=float,
                        help='how long before a complete drop (decay)')
    parser.add_argument('--decay-drop', type=float,
                        help='0 < decay_drop < 1, every epochs_drop, lr will be multiply by decay_drop')
    # -------------------- Model Type --------------------
    parser.add_argument('-n', '--name', type=str,
                        help='Name given to the model')
    parser.add_argument('--work-on', type=str,
                        help='note, beat or measure')
    parser.add_argument('--mono', default=False, action='store_true',
                        help='To work with monophonic instruments')
    # ---------- Model Id and Load ----------
    if argtype in [ArgType.ALL, ArgType.Train]:
        parser.add_argument('-m', '--model-id', type=str, default='',
                            help='The model id modelName,modelParam,nbSteps')
    elif argtype is ArgType.HPSearch:
        parser.add_argument('--model-name', type=str, default='',
                            help='The model name')
        parser.add_argument('--model-param', type=str, default='',
                            help='the model param (json file)')
        parser.add_argument('--nb-steps', type=str, default='',
                            help='Nb step to train on')

    # -------------------- Architecture --------------------
    parser.add_argument('--dropout', type=get_type(argtype, float),
                        help='Value of the dropout')
    add_store_true(parser, name='--all-sequence',
                   help='Use or not all the sequence in the RNN layer')
    add_store_true(parser, name='--lstm-state',
                   help='Use or not all the sequence in the RNN layer')
    add_store_true(parser, name='--no-sampling',
                   help='Gaussian Sampling')
    add_store_true(parser, name='--no-kld',
                   help='No KL Divergence')
    parser.add_argument('--kld-annealing-start', type=get_type(argtype, float),
                        help='Start of the annealing of the kld')
    parser.add_argument('--kld-annealing-stop', type=get_type(argtype, float),
                        help='Stop of the annealing of the kld')

    parser.set_default(
        epochs_drop=g.epochs_drop,
        decay_drop=g.decay_drop,
        name='name',
        work_on=g.work_on,
    )

    if argtype is not ArgType.HPSearch:
        parser.set_default(
            lr=g.lr,
            optimizer='adam',
            dropout=g.dropout,
            all_sequence=g.all_sequence,
            lstm_state=g.lstm_state,
            no_sampling=False,
            no_kld=False,
            kld_annealing_start=g.kld_annealing_start,
            kld_annealing_stop=g.kld_annealing_stop
        )
    else:
        parser.set_default(
            lr='1:5',
            optimizer='adam',
            dropout='0.1:0.3',
            all_sequence='False',
            lstm_state='False',
            no_sampling='False',
            no_kld='False',
            kld_annealing_start='0:0.5',
            kld_annealing_stop='0.5:1'
        )

    return parser


def add_generation_args(parser, artype=ArgType.ALL):
    """

    :param parser:
    :param artype:
    :return:
    """
    parser.add_argument('--compare-generation', default=False, action='store_true',
                        help='Compare generation after training')
    parser.add_argument('--generate', default=False, action='store_true',
                        help='Generation after training')
    parser.add_argument('--replicate', default=False, action='store_true',
                        help='Replication after training')
    parser.add_argument('--generate-fill', default=False, action='store_true',
                        help='Fill the missing instrument')
    parser.add_argument('--replicate-fill', default=False, action='store_true',
                        help='Fill the missing instrument')
    parser.add_argument('--no-duration', action='store_true', default=False,
                        help='Generate only shortest notes possible')


def add_load_model_args(parser, argtype=ArgType.ALL):
    """

    :param parser:
    :param argtype:
    :return:
    """

    parser.add_argument('-l', '--load', type=str, default='',
                        help='The name of the train model to load')
    return parser


def add_execution_type_args(parser, argtype=ArgType.ALL):
    """

    :param parser:
    :param argtype:
    :return:
    """
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='To work on a small computer with a cpu')
    parser.add_argument('--no-pc-arg', action='store_true', default=False,
                        help='To no transform parameters during pc execution')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='To set special parameters for a debug')
    parser.add_argument('--no-eager', default=False, action='store_true',
                        help='Disable eager execution')

    return parser


def add_load_data_args(parser, argtype=ArgType.ALL):
    """

    :param parser:
    :param argtype:
    :return:
    """
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_small',
                        help='The name of the data')
    return parser


def add_data_args(parser, argtype=ArgType.ALL):
    """

    :param parser:
    :param argtype:
    :return:
    """

    parser.add_argument('data', type=str, default='',
                        help='The name of the data')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    if argtype in [ArgType.ALL, ArgType.Data]:
        parser.add_argument('--length', type=str, default='',
                            help='The length of the data')
    parser.add_argument('--notes-range', type=str, default='0:88',
                        help='The length of the data')
    parser.add_argument('--instruments', type=str, default='Piano,Trombone',
                        help='The instruments considered (for space in name, put _ instead : Acoustic_Bass)')
    parser.add_argument('--bach', action='store_true', default=False,
                        help='To compute the bach data')
    parser.add_argument('--mono', action='store_true', default=False,
                        help='To compute the data where there is only one note at the same time')
    if argtype in [ArgType.ALL, ArgType.CheckData]:
        parser.add_argument('--images', action='store_true', default=False,
                            help='To also create the pianoroll')

    return parser


