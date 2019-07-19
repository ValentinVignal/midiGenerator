import argparse
import os

from src.NN.MyModel import MyModel




def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_mini',
                        help='The name of the data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=4,
                        help='The number of the batches')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('-n', '--name', type=str, default='default_name',
                        help='Name given to the model')
    # parser.add_argument('-s', '--nb-steps', type=int, default=32,
    #                     help='The number of steps used in NNetwork')
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument('-m', '--model-id', type=str, default='',
                            help='The model id modelName;modelParam;nbSteps')
    load_group.add_argument('-l', '--load', type=str, default='',
                            help='The name of the trained model to load')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')

    args = parser.parse_args()

    if args.pc:
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 1
        args.batch = 1
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    my_model = MyModel(name=args.name)
    my_model.load_data(data_transformed_path=data_transformed_path)

    args.model_id = 'pc/0' if (args.model_id == '' and args.load == '') else args.model_id
    if args.model_id != '':
        my_model.new_nn_model(model_id=args.model_id)
    elif args.load != '':
        my_model.load_model(args.load)

    my_model.train(epochs=args.epochs, batch=args.batch, verbose=1, shuffle=True)

    my_model.save_model()

    print('Done')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
