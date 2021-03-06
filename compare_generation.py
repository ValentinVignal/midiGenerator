import argparse
import os
from termcolor import cprint

from MidiGenerator.MidiGenerator import MidiGenerator


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a Midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('load', type=str, default='',
                        help='The model of the Neural Network ot load')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')
    parser.add_argument('-l', '--length', default=None,
                        help='The length of the generated music')
    parser.add_argument('--no-duration', action='store_true', default=False,
                        help='generate only shortest notes possible')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Level of verbose')

    args = parser.parse_args()

    if args.pc:
        args.length = 50
        args.seed = 2

    if not args.pc:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.length is not None:
        args.length = int(args.length)

    my_model = MidiGenerator.from_model(id=args.load)  # Load the model
    my_model.compare_generation(max_length=args.length,
                                no_duration=args.no_duration,
                                verbose=args.verbose)

    cprint('---------- Done ----------', 'grey', 'on_green')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
