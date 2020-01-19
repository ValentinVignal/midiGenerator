from src import Midi


def bayesian_opt(args):
    """
    Preprocess the args for the file bayesian-opt.py
    :param args:
    :return:
    """
    if args.pc and not args.no_pc_arg:
        args.epochs = 2
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
    if args.pc and not args.no_pc_arg:
        args.epochs = 2
        args.batch = 2
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



