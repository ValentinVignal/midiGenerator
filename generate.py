import argparse
import os

from src.NN.MyModel import MyModel


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('load', type=str, default='',
                        help='The model of the Neural Network ot load')
    parser.add_argument('--data', type=str, default='lmd_matched_mini',
                        help='The name of the data')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')
    parse.add_argument('-s', '--seed', default=10,
                       help='number of seeds or the path to the folder with the seeds')
    parser.add_argument('-l', '--length', type=int, default=200,
                        help='The length of the generated music')

    args = parser.parse_args()

    if args.pc:
        data_path = os.path.join('../Dataset', args.data)
        args.length = 50
        args.seed = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    my_model = MyModel(load_model=args.load)     # Load the model
    my_model.generate(length=args.length, seed=args.seed)

    print('Done')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
