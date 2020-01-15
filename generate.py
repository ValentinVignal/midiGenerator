import os
from termcolor import cprint

from src.MidiGenerator import MidiGenerator
from src.Args import Parser, ArgType


def main(args):
    """
        Entry point
    """
    if args.pc:
        data_path = os.path.join('../Dataset', args.data)
        args.length = 50
        args.seed = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    if not args.pc:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    my_model = MidiGenerator.with_model(id=args.load)  # Load the model
    my_model.generate_fom_data(length=args.length,
                               nb_seeds=args.nb_seeds,
                               save_images=args.images,
                               no_duration=args.no_duration)

    cprint('---------- Done ----------', 'grey', 'on_green')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    parser = Parser(argtype=ArgType.Generate)
    args = parser.parse_args()
    main(args)
