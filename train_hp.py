import argparse
import os
from termcolor import cprint, colored
import numpy as np
from pathlib import Path

from src.NN.MyModel import MyModel
from src.NN.callbacks import LossHistory


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_small',
                        help='The name of the data')
    # ----------------
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=8,
                        help='The number of the batches')
    # ----------------
    parser.add_argument('--lr', type=str, default='2:4:1',
                        help='learning rate = 10^-lr')
    parser.add_argument('-o', '--optimizer', type=str, default='adam,sgd',
                        help='Name of the optimizer (separeted with ,)')
    parser.add_argument('--epochs-drop', type=str, default='10:50:20',
                        help='how long before a complete drop (decay)')
    parser.add_argument('--decay-drop', type=str, default='0.25:0.5:0.25',
                        help='0 < decay_drop < 1, every epochs_drop, lr will be multiply by decay_drop')
    parser.add_argument('--dropout', type=str, default='0.05:0.15:0.05',
                        help='Value of the dropout')
    parser.add_argument('--type-loss', type=str, default='smooth_round',
                        help='Value of the dropout')
    parser.add_argument('--all-sequence', type=str, default='True,False',
                        help='Use or not all the sequence in the RNN layer')
    # ----------------
    parser.add_argument('-n', '--name', type=str, default='name',
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
                        help='To work on a small computer with a cpu')
    # ---------- Generation ----------
    parser.add_argument('--seed', default=4,
                        help='number of seeds or the path to the folder with the seeds')
    parser.add_argument('--length', type=int, default=300,
                        help='The length of the generated music')
    parser.add_argument('--images', action='store_true', default=False,
                        help='Save the images for each instruments')
    parser.add_argument('--no-duration', action='store_true', default=False,
                        help='Generate only shortest notes possible')
    parser.add_argument('--verbose_generation', type=int, default=1,
                        help='Level of verbose')

    args = parser.parse_args()

    # ----- pc -----
    if args.pc:
        # args.data = 'lmd_matched_mini'
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 2
        args.batch = 1
        args.seed = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    def create_list(string):
        string_list = string.split(':')
        if len(string_list) == 1:
            return [float(string_list[0])]
        else:
            return list(np.arange(float(string_list[0]), float(string_list[1]) + float(string_list[2]), float(string_list[2])))

    # ---------- Training ----------
    # ----- lr -----
    args.lr = create_list(args.lr)
    args.lr = list(map(lambda x: 10 ** (-x), args.lr))
    # ----- optimizer -----
    args.optimizer = args.optimizer.split(',')
    # ----- Epochs Drop -----
    args.epochs_drop = create_list(args.epochs_drop)
    # ----- Decay Drop -----
    args.decay_drop = create_list(args.decay_drop)
    # ----- Dropout -----
    args.dropout = create_list(args.dropout)
    # ----- Type Loss -----
    args.type_loss = args.type_loss.split(',')
    # ----- All Sequence -----
    args.all_sequence = list(map(lambda x: x == 'True', args.all_sequence.split(',')))

    # ---------- Generation ----------
    args.images = True
    args.no_duration = True

    # -------------------- Make the directory for the results --------------------
    test_path = Path('tests_hp')
    id = 0
    while (test_path / 'test_{0}'.format(id)).exists():
        id += 1
    test_path = test_path / 'test_{0}'.format(id)
    test_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------
    loss_history = LossHistory()

    # ------------------------------------------------
    i = 1
    nb_tests = len(args.lr) * len(args.optimizer) * len(args.epochs_drop) * len(args.decay_drop) * len(
        args.dropout) * len(args.type_loss) * len(args.all_sequence)
    for lr in args.lr:
        for optimizer in args.optimizer:
            for epochs_drop in args.epochs_drop:
                for decay_drop in args.decay_drop:
                    for dropout in args.dropout:
                        for type_loss in args.type_loss:
                            for all_sequence in args.all_sequence:
                                cprint('Test {0}/{1}'.format(i, nb_tests), 'yellow', 'on_blue')
                                print('lr :', colored(lr, 'magenta'),
                                      '- optimizer :', colored(optimizer, 'magenta'),
                                      '- epochs_drop :', colored(epochs_drop, 'magenta'),
                                      '- decay_drop :', colored(decay_drop, 'magenta'),
                                      '- dropout :', colored(dropout, 'magenta'),
                                      '- type_loss :', colored(type_loss, 'magenta'),
                                      '- all_sequence :', colored(all_sequence, 'magenta'))

                                my_model = MyModel(name=args.name)
                                my_model.load_data(data_transformed_path=data_transformed_path)
                                # Choose GPU
                                if not args.pc:
                                    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

                                if args.model_id != '':
                                    opt_param = {
                                        'lr': lr,
                                        'name': optimizer,
                                        'drop': decay_drop,
                                        'epoch_drop': epochs_drop
                                    }
                                    my_model.new_nn_model(model_id=args.model_id,
                                                          opt_param=opt_param,
                                                          dropout=dropout,
                                                          type_loss=type_loss,
                                                          all_sequence=all_sequence,
                                                          print_model=False)

                                my_model.train(epochs=args.epochs, batch=args.batch, callbacks=[loss_history],
                                               verbose=1)

                                path = my_model.save_model()
                                hparams = {
                                    'index': i,
                                    'lr': lr,
                                    'optimizer': optimizer,
                                    'epochs_drop': epochs_drop,
                                    'decay_drop': decay_drop,
                                    'dropout': dropout,
                                    'type_loss': type_loss,
                                    'all_sequence': all_sequence
                                }
                                loss_history.paths.append(path)
                                loss_history.hparams.append(hparams)

                                cprint(
                                    'Best loss for now : {0}'.format(
                                        loss_history.logs[loss_history.best_index]['loss']), 'yellow')
                                my_model.generate(length=args.length,
                                                  seed=args.seed,
                                                  save_images=args.images,
                                                  no_duration=args.no_duration,
                                                  verbose=args.verbose_generation)

                                del my_model
                                i += 1
    loss_history.save_summary()

    cprint('---------- Done ----------', 'grey', 'on_green')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
