import argparse
import os

from src.NN.MyModel import MyModel




def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset')
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_mini', metavar='N',
                        help='The name of the data')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('-b', '--batch', type=int, default=1,
                        help='The number of the batchs')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    #parser.add_argument('--log-interval', type=int, default=5, metavar='N',
    #                    help='how many batch to wait before logging training status')
    parser.add_argument('-n', '--name', type=str, default='default_name',
                        help='how many batch to wait before logging training status')
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument('-m', '--model', type=str, default='18',
                            help='The model of the Neural Network used for the interpolation')
    load_group.add_argument('-l', '--load', type=str, default='',
                            help='The name of the trained model to load')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')

    args = parser.parse_args()

    if args.pc:
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    my_model = MyModel()
    my_model.load_data(data_transformed_path=data_transformed_path)

    if args.model != '':
        nb_steps = int(args.model)
        my_model.new_nn_model(nb_steps=nb_steps)
    elif args.load != '':
        my_model.load_model(args.load)

    my_model.train(epochs=args.epochs, batch=args.batch, verbose=1, shuffle=True)

    my_model.save_model()
    my_model.print_weights()

    print('Done')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
