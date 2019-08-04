import argparse
import os

from src.NN.MyModel import MyModel




def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_small',
                        help='The name of the data')
    # ----------------
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=4,
                        help='The number of the batches')
    # ----------------
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='Name of the optimizer')
    # ----------------
    parser.add_argument('-n', '--name', type=str, default='defaultName',
                        help='Name given to the model')
    # ----------------
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument('-m', '--model-id', type=str, default='',
                            help='The model id modelName,modelParam,nbSteps')
    load_group.add_argument('-l', '--load', type=str, default='',
                            help='The name of the trained model to load')
    # ----------------
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')

    args = parser.parse_args()

    if args.pc:
        args.data = 'lmd_matched_small'
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 1
        args.batch = 1
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    my_model = MyModel(name=args.name)
    my_model.load_data(data_transformed_path=data_transformed_path)
    # Choose GPU
    if not args.pc:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.model_id = 'pc;0;8' if (args.model_id == '' and args.load == '') else args.model_id
    if args.model_id != '':
        opt_param = {
            'lr': args.lr,
            'name': args.optimizer
        }

        my_model.new_nn_model(model_id=args.model_id, opt_param=opt_param)
    elif args.load != '':
        my_model.load_model(args.load)

    my_model.train(epochs=args.epochs, batch=args.batch)

    my_model.save_model()

    print('Done')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
