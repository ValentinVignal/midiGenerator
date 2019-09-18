import argparse
import os
from termcolor import cprint

from src.NN.MyModel import MyModel
import src.global_variables as g


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_small',
                        help='The name of the data')
    # ----------------
    parser.add_argument('-e', '--epochs', type=int, default=g.epochs,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=4,
                        help='The number of the batches')
    # ---------------- Optimizer ----------------
    parser.add_argument('--lr', type=float, default=g.lr,
                        help='learning rate')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='Name of the optimizer')
    parser.add_argument('--epochs-drop', type=float, default=g.epochs_drop,
                        help='how long before a complete drop (decay)')
    parser.add_argument('--decay-drop', type=float, default=g.decay_drop,
                        help='0 < decay_drop < 1, every epochs_drop, lr will be multiply by decay_drop')
    parser.add_argument('--type-loss', type=str, default=g.type_loss,
                        help='Value of the dropout')
    parser.add_argument('--lambdas-loss', type=str, default=g.lambdas_loss,
                        help='Value of the lambdas (activation and duration) for the loss function')
    # ---------------- Neural Network Model ----------------
    parser.add_argument('--dropout', type=float, default=g.dropout,
                        help='Value of the dropout')
    parser.add_argument('--all-sequence', default=False, action='store_true',
                        help='Use or not all the sequence in the RNN layer')
    parser.add_argument('--lstm-state', default=False, action='store_true',
                        help='Use or not all the sequence in the RNN layer')
    # ---------------- Batch Norm ----------------
    parser.add_argument('--no-batch-norm', default=False, action='store_true',
                        help='Either to use batch norm')
    parser.add_argument('--bn-momentum', type=float, default=g.bn_momentum,
                        help='The value of the momentum of the batch normalization layers')
    # ---------------- Training options ----------------
    parser.add_argument('--noise', type=float, default=g.noise,
                        help='If not 0, add noise to the input for training')
    # ----------------
    parser.add_argument('-n', '--name', type=str, default='name',
                        help='Name given to the model')
    parser.add_argument('--work-on', type=str, default=g.work_on,
                        help='note, beat or measure')
    # ----------------
    parser.add_argument('--evaluate', default=False, action='store_true',
                        help='Evaluate the model after the training')
    parser.add_argument('--compare-generation', default=False, action='store_true',
                        help='Compare generation after training')
    parser.add_argument('--generate', default=False, action='store_true',
                        help='Generation after training')
    parser.add_argument('--no-duration', action='store_true', default=False,
                        help='Generate only shortest notes possible')
    parser.add_argument('--check-batch', type=int, default=-1,
                        help='Batch to check')
    # ----------------
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument('-m', '--model-id', type=str, default='',
                            help='The model id modelName,modelParam,nbSteps')
    load_group.add_argument('-l', '--load', type=str, default='',
                            help='The name of the trained model to load')
    # ---------------- Hardware options ----------------
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='To work on a small computer with a cpu')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='To set special paremeters for a debug')

    args = parser.parse_args()

    if args.pc:
        # args.data = 'lmd_matched_mini'
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 15
        args.batch = 3
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    # -------------------- Create model --------------------
    my_model = MyModel(name=args.name)
    # Choose GPU
    if not args.pc:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.debug:
        pass

    if args.model_id != '':
        my_model.load_data(data_transformed_path=data_transformed_path)
        opt_param = {
            'lr': args.lr,
            'name': args.optimizer,
            'drop': float(args.decay_drop),
            'epoch_drop': float(args.epochs_drop)
        }
        model_options = {
            'dropout': args.dropout,
            'all_sequence': args.all_sequence,
            'lstm_state': args.lstm_state,
            'no_batch_norm': args.no_batch_norm,
            'bn_momentum': args.bn_momentum,
            'lambdas_loss': args.lambdas_loss
        }
        my_model.new_nn_model(model_id=args.model_id,
                              opt_param=opt_param,
                              work_on=args.work_on,
                              type_loss=args.type_loss,
                              model_options=model_options)
    elif args.load != '':
        my_model.recreate_model(args.load)

    # -------------------- Train --------------------
    my_model.train(epochs=args.epochs, batch=args.batch, noise=args.noise)

    # -------------------- Test --------------------
    if args.evaluate:
        my_model.evaluate()

    # -------------------- Test overfit --------------------
    if args.compare_generation:
        my_model.compare_generation(max_length=None,
                                    no_duration=args.no_duration,
                                    verbose=1)

    # -------------------- Generate --------------------
    if args.generate:
        my_model.generate_fom_data(nb_seeds=4, save_images=True, no_duration=args.no_duration)

    # -------------------- Debug batch generation --------------------
    if args.check_batch > -1:
        for i in range(len(my_model.my_sequence)):
            my_model.compare_test_predict_on_batch(i)

    # -------------------- Save the model --------------------
    my_model.save_model()

    cprint('---------- Done ----------', 'grey', 'on_green')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
