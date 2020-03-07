from epicpath import EPath
import warnings
import pickle

from src import Midi


def bayesian_opt(args):
    """
    Preprocess the args for the file bayesian-opt.py
    :param args:
    :return:
    """
    if args.pc and not args.no_pc_arg:
        args.epochs = 1
        args.n_calls = 2
    if args.in_place and args.from_checkpoint is None:
        warnings.warn('The arg "in-place" is set to "True" while the arg "from-checkpoint" is "None"')
    if args.from_checkpoint is not None:
        # It means the bayesian optimization continues a previous one, hence, some args must be the same the
        # the optimization stays coherent
        with open(EPath('hp_search', f'bayesian_opt_{"_".join([str(s) for s in args.from_checkpoint.split("-")])}', 'checkpoint', 'args.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            saved_args = d['args']
            for k, value in vars(saved_args).items():
                # if k not in ['in_place', 'from_checkpoint', 'n_calls', 'gpu', 'debug']:
                if k not in ['from_checkpoint']:
                    setattr(args, k, value)

    return args


def train(args):
    """
    For the file train.py
    :param args:
    :return:
    """
    if args.mono:
        if args.loss_name == 'basic':
            args.loss_name = 'mono'
        elif 'mono' not in args.loss_name:
            args.loss_name = 'mono_' + args.loss_name
    if args.song_number == -1:
        args.song_number = None
    if args.pc and not args.no_pc_arg:
        args.epochs = 10
        args.batch = 2
        args.max_queue_size = 10        # Default parameter of Keras
        args.workers = 4
    return args


def check_dataset(args):
    """
    For the file check_dataset.py
    :param args:
    :return:
    """
    # note_range
    s = args.notes_range.split(':')
    args.notes_range = (int(s[0]), int(s[1]))
    # instruments
    args.instruments = list(map(lambda instrument: ' '.join(instrument.split('_')),
                                args.instruments.split(',')))
    # bach
    if args.bach:
        args.instruments = Midi.instruments.bach_instruments

    return args


def compute_data(args):
    """
    For the file compute_data.py
    :param args:
    :return:
    """
    # length
    args.length = None if args.length == '' else int(args.length)
    # note range
    s = args.notes_range.split(':')
    args.notes_range = (int(s[0]), int(s[1]))
    # instruments
    args.instruments = list(map(lambda instrument: ' '.join(instrument.split('_')),
                                args.instruments.split(',')))
    # bach
    if args.bach:
        args.instruments = Midi.inst.bach_instruments

    return args


def generate(args):
    """
    For the file generate.py
    :param args:
    :return:
    """
    if args.pc and not args.no_pc_arg:
        args.length = 50
        args.seed = 2
    return args


def clean(args):
    """

    :param args:
    :return:
    """
    return args


def zip(args):
    """

    :param args:
    :return:
    """
    if not args.midi and not args.hp and not args.model:
        args.all = True
    else:
        args.all = False
    return args


def hp_summary(args):
    """

    :param args:
    :return:
    """
    args.folder = EPath('hp_search', f'bayesian_opt_{args.folder}')
    return args


def n_scripts_bo(args):
    """

    :param args:
    :return:
    """
    return args



