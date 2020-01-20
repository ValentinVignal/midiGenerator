import argparse
import os
from termcolor import cprint, colored
import numpy as np
import tensorflow as tf
K = tf.keras.backend

from MidiGenerator.MidiGenerator import MidiGenerator
from Callbacks.LossHistory import LossHistory
import src.global_variables as g



def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a Midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_small',
                        help='The name of the data')
    # ----------------
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('-b', '--batch', type=int, default=8,
                        help='The number of the batches')
    # ----------------
    parser.add_argument('--lr', type=str, default='2:4:1',
                        help='learning rate = 10^-lr')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='Name of the optimizer (separeted with ,)(ex : adam,sgd)')
    parser.add_argument('--epochs-drop', type=str, default='50:100:50',
                        help='how long before a complete drop (decay)')
    parser.add_argument('--decay-drop', type=str, default='0.25:0.5:0.25',
                        help='0 < decay_drop < 1, every epochs_drop, lr will be multiply by decay_drop')
    parser.add_argument('--dropout', type=str, default='0.1:0.2:0.1',
                        help='Value of the dropout')
    parser.add_argument('--type-loss', type=str, default=g.type_loss,
                        help='Value of the dropout')
    parser.add_argument('--all-sequence', type=str, default='False',
                        help='Use or not all the sequence in the RNN layer (separated with ,)')
    parser.add_argument('--lstm-state', type=str, default='False',
                        help='Use or not all the sequence in the RNN layer (separated with ,)')
    parser.add_argument('--work-on', type=str, default=g.work_on,
                        help='note, beat or measure')
    # ----------------
    parser.add_argument('-n', '--name', type=str, default='name',
                        help='Name given to the model')
    # ----------------
    parser.add_argument('--model-name', type=str, default='rnn',
                            help='The model name')
    parser.add_argument('--model-param', type=str, default='pc,0,1',
                        help='the model param (json file)')
    parser.add_argument('--nb-steps', type=str, default='8,16',
                        help='Nb step to train on')
    # ----------------
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='To work on a small computer with a cpu')
    parser.add_argument('--mono', action='store_true', default=False,
                        help='To work with monophonic instruments')
    # ---------- Generation ----------
    parser.add_argument('--compare-generation', default=False, action='store_true',
                        help='Compare generation after training')
    parser.add_argument('--generate', default=False, action='store_true',
                        help='Generation after training')
    parser.add_argument('--seed', default=4,
                        help='number of seeds or the path to the folder with the seeds')
    parser.add_argument('--length', type=int, default=20,
                        help='The length of the generated music')
    parser.add_argument('--no-duration', action='store_true', default=False,
                        help='Generate only shortest notes possible')
    parser.add_argument('--verbose_generation', type=int, default=1,
                        help='Level of verbose')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Fraction of the training data to be used as validation data')

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
    if args.mono:
        data_transformed_path += 'Mono'

    # Choose GPU
    if not args.pc:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
    # LSTM State
    args.lstm_state = list(map(lambda x: x == 'True', args.lstm_state.split(',')))
    # Model Param
    args.model_param = args.model_param.split(',')
    # Nb Steps
    args.nb_steps = args.nb_steps.split(',')

    # ---------- Generation ----------
    args.images = True
    args.no_duration = True

    # --------------------------------------------
    loss_history = LossHistory()

    # ------------------------------------------------
    all_params = []
    for lr in args.lr:
        for optimizer in args.optimizer:
            for epochs_drop in args.epochs_drop:
                for decay_drop in args.decay_drop:
                    for dropout in args.dropout:
                        for type_loss in args.type_loss:
                            for all_sequence in args.all_sequence:
                                for lstm_state in args.lstm_state:
                                    for model_param in args.model_param:
                                        for nb_steps in args.nb_steps:
                                            all_params.append({
                                                'lr': lr,
                                                'optimizer': optimizer,
                                                'epochs_drop': epochs_drop,
                                                'decay_drop': decay_drop,
                                                'dropout': dropout,
                                                'type_loss': type_loss,
                                                'all_sequence': all_sequence,
                                                'lstm_state': lstm_state,
                                                'model_id': args.model_name + ',' + model_param + ',' + nb_steps,
                                            })

    for index, params in enumerate(all_params):
        my_model = MidiGenerator(name=args.name)
        my_model.load_data(data_transformed_path=data_transformed_path)
        cprint('Test {0}/{1}'.format(index + 1, len(all_params)), 'yellow', 'on_blue')
        print('lr :', colored(params['lr'], 'magenta'),
              '- optimizer :', colored(params['optimizer'], 'magenta'),
              '- epochs_drop :', colored(params['epochs_drop'], 'magenta'),
              '- decay_drop :', colored(params['decay_drop'], 'magenta'),
              '- dropout :', colored(params['dropout'], 'magenta'),
              '- type_loss :', colored(params['type_loss'], 'magenta'),
              '- all_sequence :', colored(params['all_sequence'], 'magenta'),
              '- lstm_state :', colored(params['lstm_state'], 'magenta'),
              '- model_id :', colored(params['model_id'], 'magenta'))

        opt_param = {
            'lr': params['lr'],
            'name': params['optimizer'],
            'drop': params['decay_drop'],
            'epoch_drop': params['epochs_drop']
        }
        model_options = {
            'dropout': params['dropout'],
            'all_sequence': params['all_sequence'],
            'lstm_state': params['lstm_state'],
        }
        my_model.new_nn_model(model_id=params['model_id'],
                              work_on=args.work_on,
                              opt_param=opt_param,
                              type_loss=params['type_loss'],
                              model_options=model_options,
                              print_model=False)

        my_model.train(epochs=args.epochs, batch=args.batch, callbacks=[loss_history],
                       verbose=1, validation=args.validation)

        path = my_model.save_model()
        hparams = params
        hparams['index'] = index
        loss_history.paths.append(path)
        loss_history.hparams.append(hparams)
        loss_history.update_summary(index)

        cprint(
            'Best loss for now : {0}'.format(
                loss_history.logs[loss_history.best_index]['loss']), 'yellow')
        if args.compare_generation:
            my_model.compare_generation(max_length=None,
                                        no_duration=args.no_duration,
                                        verbose=1)

        if args.generate:
            my_model.generate_from_data(length=args.length,
                                        nb_seeds=args.seed,
                                        save_images=True,
                                        no_duration=args.no_duration,
                                        verbose=args.verbose_generation)
        K.clear_session()
        del my_model.my_sequence
        del my_model.keras_nn
        del my_model

    loss_history.update_best_summary()

    cprint('---------- Done ----------', 'grey', 'on_green')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
