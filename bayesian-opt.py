"""
File to do an Bayesian Optimization of the hyper parameters
"""
# Import
import os
from termcolor import cprint, colored
import tensorflow as tf
import gc
import skopt
from skopt import gp_minimize
import pickle
# Personal import
from src.text import summary
from src.MidiGenerator import MidiGenerator
from src import Args
from src.Args import ArgType, Parser
from src.NN import Sequences
from src.NN.KerasNeuralNetwork import KerasNeuralNetwork
from src.NN import Callbacks
from src import BayesianOpt as BO
from src.BayesianOpt.process_args import string_to_tuple, ten_power, string_to_bool

# Variables
K = tf.keras.backend

os.system('echo start bayesian-opt.py')

# ------------------------------------------------------------
# Functions to create dimensions from args
# ------------------------------------------------------------


def create_dimensions(args):
    """
    From the args, it creates all the dimensions on which the bayesian optimization will work on

    :param args:
    :return:
    """

    dimensions = BO.Dimensions()

    # lr
    lr_tuple = string_to_tuple(args.lr, t=ten_power)
    dimensions.add_Real(lr_tuple, name='lr', prior='log-uniform')
    # optimizer
    opt_tuple = string_to_tuple(args.optimizer, t=str, separator=',')
    dimensions.add_Categorical(opt_tuple, name='optimizer')
    # decay
    decay_tuple = string_to_tuple(args.decay, t=ten_power)
    dimensions.add_Real(decay_tuple, name='decay', prior='log-uniform')
    # dropout d
    dropout_d_tuple = string_to_tuple(args.dropout_d)
    dimensions.add_Real(dropout_d_tuple, name='dropout_d')
    # dropout c
    dropout_c_tuple = string_to_tuple(args.dropout_c)
    dimensions.add_Real(dropout_c_tuple, name='dropout_c')
    # dropout r
    dropout_r_tuple = string_to_tuple(args.dropout_r)
    dimensions.add_Real(dropout_r_tuple, name='dropout_r')
    # sampling
    sampling_tuple = string_to_tuple(args.no_sampling, t=lambda x: not string_to_bool(x), separator=',')
    dimensions.add_Categorical(sampling_tuple, name='sampling')
    # kld
    kld_tuple = string_to_tuple(args.no_kld, t=lambda x: not string_to_bool(x), separator=',')
    dimensions.add_Categorical(kld_tuple, name='kld')
    # all sequence
    all_sequence_tuple = string_to_tuple(args.all_sequence, t=string_to_bool, separator=',')
    dimensions.add_Categorical(all_sequence_tuple, name='all_sequence')
    # model name
    model_name_tuple = string_to_tuple(args.model_name, t=str, separator=',')
    dimensions.add_Categorical(model_name_tuple, 'model_name')
    # model param
    model_param_tuple = string_to_tuple(args.model_param, t=str, separator=',')
    dimensions.add_Categorical(model_param_tuple, 'model_param')
    # nb steps
    nb_steps_tuple = string_to_tuple(args.nb_steps, t=int, separator=',')
    dimensions.add_Categorical(nb_steps_tuple, 'nb_steps')
    # kld annealing start
    kld_annealing_start_tuple = string_to_tuple(args.kld_annealing_start)
    dimensions.add_Real(kld_annealing_start_tuple, name='kld_annealing_start')
    # kld annealing stop
    kld_annealing_stop_tuple = string_to_tuple(args.kld_annealing_stop)
    dimensions.add_Real(kld_annealing_stop_tuple, name='kld_annealing_stop')
    # kld sum
    kld_sum_tuple = string_to_tuple(args.no_kld_sum, t=lambda x: not string_to_bool(x), separator=',')
    dimensions.add_Categorical(kld_sum_tuple, name='kld_sum')
    # loss name
    loss_name_tuple = string_to_tuple(args.loss_name, t=str, separator=',')
    dimensions.add_Categorical(loss_name_tuple, name='loss_name')
    # lambda scale
    l_scale_tuple = string_to_tuple(args.l_scale, t=ten_power, separator=':')
    dimensions.add_Real(l_scale_tuple, 'l_scale', prior='log-uniform')
    # lambda rhythm
    l_rhythm_tuple = string_to_tuple(args.l_rhythm, t=ten_power, separator=':')
    dimensions.add_Real(l_rhythm_tuple, name='l_rhythm', prior='log-uniform')
    # No all step rhythm
    take_all_step_rhythm_tuple = string_to_tuple(args.no_all_step_rhythm, t=lambda x: not string_to_bool(x),
                                                 separator=',')
    dimensions.add_Categorical(take_all_step_rhythm_tuple, name='take_all_step_rhythm')
    # sah
    sah_tuple = string_to_tuple(args.sah, t=string_to_bool, separator=',')
    dimensions.add_Categorical(sah_tuple, name='sah')
    # lambda semitone
    l_semitone_tuple = string_to_tuple(args.l_semitone, t=ten_power, separator=':')
    dimensions.add_Real(l_semitone_tuple, name='l_semitone', prior='log-uniform')
    # lambda tone
    l_tone_tuple = string_to_tuple(args.l_tone, t=ten_power, separator=':')
    dimensions.add_Real(l_tone_tuple, name='l_tone', prior='log-uniform')
    # lambda tritone
    l_tritone_tuple = string_to_tuple(args.l_tritone, t=ten_power, separator=':')
    dimensions.add_Real(l_tritone_tuple, name='l_tritone', prior='log-uniform')
    # rpoe
    rpoe_tuple = string_to_tuple(args.no_rpoe, t=lambda x: not string_to_bool(x), separator=',')
    dimensions.add_Categorical(rpoe_tuple, name='rpoe')

    return dimensions


def get_history_acc(history):
    """

    :param history:
    :return: The total validation accuracy
    """
    accuracy = 0
    i = 0
    while f'val_Output_{i}_acc' in history:
        accuracy += history[f'val_Output_{i}_acc'][-1]
        i += 1
    return accuracy / i


def str_hp_to_print(name, value, exp_format=False, first_printed=False):
    """

    :param name:
    :param value:
    :param exp_format:
    :param first_printed:
    :return: string which is pretty to print (to show current hp tested)
    """
    s = ''
    if not first_printed:
        s += ' - '
    s += f'{name}: '
    if exp_format:
        s += colored(f'{value:.1e}', 'magenta')
    else:
        s += colored(f'{value}', 'magenta')
    return s


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
#                                       Main function
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def main(args):
    """
        Entry point
    """
    # -------------------- Setup --------------------
    # ----- pc -----
    if args.pc:
        # args.data = 'lmd_matched_mini'
        data_path = os.path.join('../Dataset', args.data)
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'
    if args.mono:
        data_transformed_path += 'Mono'

    # Choose GPU
    if not args.pc:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.seq2np:
        KerasNeuralNetwork.slow_down_cpu(
            nb_inter=args.nb_inter_threads,
            nb_intra=args.nb_intra_threads
        )

    # get x and y if args.seq2np
    if args.seq2np:
        x_dict, y_dict = {}, {}

    from_checkpoint = args.from_checkpoint is not None
    saved_checkpoint_folder = None
    id = None
    if from_checkpoint:
        saved_checkpoint_folder = BO.save.get_folder_path(args.from_checkpoint) / 'checkpoint'
        id = args.from_checkpoint if args.in_place else None

    folder_path = BO.save.get_folder_path(id=id, name=args.bo_name)
    folder_path.mkdir(parents=True, exist_ok=args.in_place)  # We want it to act as a token
    checkpoint_folder = folder_path / 'checkpoint'
    checkpoint_folder.mkdir(exist_ok=args.in_place)

    # -------------------- Setup Bayesian Optimization --------------------
    saved_checkpoint, dimensions = None, None

    global best_accuracy
    global iteration
    global max_iterations
    if not from_checkpoint:
        best_accuracy = 0
        iteration = 0
    else:
        saved_checkpoint, dimensions = BO.load.from_checkpoint(saved_checkpoint_folder)
        best_accuracy = - saved_checkpoint.fun      # Because skopt minimizes it
        iteration = saved_checkpoint.func_vals.size
        print('Checkpoint loaded from', colored(f'{saved_checkpoint_folder.parent}', 'green'),
              'save in place:', colored(f'{args.in_place}', 'yellow'))
    max_iterations = args.n_calls + iteration

    dimensions = create_dimensions(args) if dimensions is None else dimensions

    def create_model(lr, optimizer, decay, dropout_d, dropout_c, dropout_r, sampling, kld, all_sequence, model_name,
                     model_param, nb_steps, kld_annealing_start, kld_annealing_stop, kld_sum, loss_name, l_scale,
                     l_rhythm, take_all_step_rhythm, sah, l_semitone, l_tone, l_tritone, rpoe):
        """
        Creates a model from all the inputs == dimensions of the bayesian optimization
        """
        midi_generator = MidiGenerator(name=args.name)
        midi_generator.load_data(data_transformed_path=data_transformed_path, verbose=0)

        model_id = f'{model_name},{model_param},{nb_steps}'

        opt_params = dict(
            lr=lr,
            name=optimizer,
            decay_drop=float(args.decay_drop),
            epoch_drop=float(args.epochs_drop),
            decay=decay
        )
        model_options = dict(
            dropout_d=dropout_d,
            dropout_c=dropout_c,
            dropout_r=dropout_r,
            all_sequence=all_sequence,
            lstm_state=args.lstm_state,
            sampling=sampling,
            kld=kld,
            kld_annealing_start=kld_annealing_start,
            kld_annealing_stop=kld_annealing_stop,
            kld_sum=kld_sum,
            sah=sah,
            rpoe=rpoe
        )

        loss_options = dict(
            loss_name=loss_name,
            l_scale=l_scale,
            l_rhythm=l_rhythm,
            take_all_step_rhythm=take_all_step_rhythm,
            l_semitone=l_semitone,
            l_tone=l_tone,
            l_tritone=l_tritone
        )

        midi_generator.new_nn_model(
            model_id=model_id,
            opt_param=opt_params,
            work_on=args.work_on,
            model_options=model_options,
            loss_options=loss_options,
            print_model=False,
            predict_offset=args.predict_offset
        )
        return midi_generator

    def fitness(l):
        """
        From all the inputs == dimensions of the bayesian optimization, it creates a model, train it, add return
        its negative accuracy (skopt works by minimizing a function)
        """
        global iteration
        global max_iterations
        iteration += 1

        # ---------------------------------------- Get the variables ----------------------------------------
        lr = dimensions.get_value_param('lr', l)
        optimizer = dimensions.get_value_param('optimizer', l)
        decay = dimensions.get_value_param('decay', l)
        dropout_d = dimensions.get_value_param('dropout_d', l)
        dropout_c = dimensions.get_value_param('dropout_c', l)
        dropout_r = dimensions.get_value_param('dropout_r', l)
        sampling = dimensions.get_value_param('sampling', l)
        kld = dimensions.get_value_param('kld', l)
        all_sequence = dimensions.get_value_param('all_sequence', l)
        model_name = dimensions.get_value_param('model_name', l)
        model_param = dimensions.get_value_param('model_param', l)
        nb_steps = dimensions.get_value_param('nb_steps', l)
        kld_annealing_start = dimensions.get_value_param('kld_annealing_start', l)
        kld_annealing_stop = dimensions.get_value_param('kld_annealing_stop', l)
        kld_sum = dimensions.get_value_param('kld_sum', l)
        loss_name = dimensions.get_value_param('loss_name', l)
        l_scale = dimensions.get_value_param('l_scale', l)
        l_rhythm = dimensions.get_value_param('l_rhythm', l)
        take_all_step_rhythm = dimensions.get_value_param('take_all_step_rhythm', l)
        sah = dimensions.get_value_param('sah', l)
        l_semitone = dimensions.get_value_param('l_semitone', l)
        l_tone = dimensions.get_value_param('l_tone', l)
        l_tritone = dimensions.get_value_param('l_tritone', l)
        rpoe = dimensions.get_value_param('rpoe', l)

        # ------------------------------ Print the information to the user ------------------------------
        s = 'Iteration ' + colored(f'{iteration}/{max_iterations}', 'yellow')
        model_id = f'{model_name},{model_param},{nb_steps}'
        s += str_hp_to_print('model', model_id, first_printed=False)
        s += str_hp_to_print('lr', lr, exp_format=True)
        s += str_hp_to_print('opt', optimizer)
        s += str_hp_to_print('decay', decay, exp_format=True)
        s += str_hp_to_print('dropout_d', dropout_d, exp_format=True)
        s += str_hp_to_print('dropout_c', dropout_c, exp_format=True)
        s += str_hp_to_print('dropout_r', dropout_r, exp_format=True)
        s += str_hp_to_print('sampling', sampling)
        s += str_hp_to_print('kld', kld)
        s += str_hp_to_print('all_sequence', all_sequence)
        s += str_hp_to_print('kld_annealing_start', kld_annealing_start, exp_format=True)
        s += str_hp_to_print('kld_annealing_stop', kld_annealing_stop, exp_format=True)
        s += str_hp_to_print('kld_sum', kld_sum)
        s += str_hp_to_print('loss_name', loss_name)
        s += str_hp_to_print('l_scale', l_scale, exp_format=True)
        s += str_hp_to_print('l_rhythm', l_rhythm, exp_format=True)
        s += str_hp_to_print('take_all_step_rhythm', take_all_step_rhythm)
        s += str_hp_to_print('sah', sah)
        s += str_hp_to_print('l_semitone', l_semitone, exp_format=True)
        s += str_hp_to_print('l_tone', l_tone, exp_format=True)
        s += str_hp_to_print('l_tritone', l_tritone, exp_format=True)
        s += str_hp_to_print('rpoe', rpoe)
        print(s)

        # -------------------- Create the model --------------------
        midi_generator = create_model(
            lr=lr,
            optimizer=optimizer,
            decay=decay,
            dropout_d=dropout_d,
            dropout_c=dropout_c,
            dropout_r=dropout_r,
            sampling=sampling,
            kld=kld,
            all_sequence=all_sequence,
            model_name=model_name,
            model_param=model_param,
            nb_steps=nb_steps,
            kld_annealing_start=kld_annealing_start,
            kld_annealing_stop=kld_annealing_stop,
            kld_sum=kld_sum,
            loss_name=loss_name,
            l_scale=l_scale,
            l_rhythm=l_rhythm,
            take_all_step_rhythm=take_all_step_rhythm,
            sah=sah,
            l_semitone=l_semitone,
            l_tone=l_tone,
            l_tritone=l_tritone,
            rpoe=rpoe
        )

        # -------------------- Train the model --------------------
        best_accuracy_callback = Callbacks.BestAccuracy()
        if args.seq2np:
            # get x and y
            if nb_steps not in x_dict or nb_steps not in y_dict:
                midi_generator.get_sequence(
                    path=midi_generator.data_transformed_path.as_posix(),
                    nb_steps=midi_generator.nb_steps,
                    batch_size=args.batch,
                    work_on=midi_generator.work_on
                )
                x, y = Sequences.sequence_to_numpy(midi_generator.sequence)
                x_dict[nb_steps] = x
                y_dict[nb_steps] = y
                del x, y
            # Train
            history = midi_generator.keras_nn.train(epochs=args.epochs, x=x_dict[nb_steps], y=y_dict[nb_steps],
                                                    verbose=1, validation=args.validation,
                                                    callbacks=[best_accuracy_callback], batch_size=args.batch)
        else:
            history = midi_generator.train(epochs=args.epochs, batch=args.batch, callbacks=[best_accuracy_callback],
                                           verbose=1, validation=args.validation, fast_sequence=args.fast_seq,
                                           memory_sequence=args.memory_seq)
        accuracy = best_accuracy_callback.best_acc

        global best_accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Print accuracy to the user
        print(f'Accuracy:',
              colored(f'{accuracy:.2%}', 'cyan'),
              '- Best Accuracy for now:',
              colored(f'{best_accuracy:.2%}', 'white', 'on_blue'),
              '\n')

        midi_generator.keras_nn.clear_session()
        del midi_generator
        del history
        gc.collect()

        '''
        # To do quick tests
        res = lr * decay / dropout ** (2 + nb_steps)
        print(res)
        return res
        '''

        # Return negative accuracy because skopt works by minimizing functions
        return -accuracy

    # ------------------------------------------------------------
    #               Actually run the Bayesian search
    # ------------------------------------------------------------
    dimensions.save(checkpoint_folder / 'dimensions.p')

    with open(checkpoint_folder / 'args.p', 'wb') as dump_file:
        pickle.dump(dict(args=args), dump_file)

    summary.summarize(
        # Function parameters
        path=folder_path,
        title='Args of bayesian optimization',
        # Summary params
        **vars(args)
    )
    checkpoint_callback = skopt.callbacks.CheckpointSaver(
        checkpoint_path=(checkpoint_folder / 'search_result.pkl').path,
        store_objective=False
    )

    n_random_starts = 10
    if args.pc and not args.no_pc_arg:
        n_random_starts = 1
    if from_checkpoint:
        n_random_starts = 0

    search_result = gp_minimize(
        func=fitness,
        dimensions=dimensions.dimensions,
        acq_func='EI',
        n_calls=args.n_calls,
        x0=dimensions.default_dim,
        callback=[checkpoint_callback],
        n_random_starts=n_random_starts
    )

    BO.save.save_search_result(
        search_result=search_result,
        dimensions=dimensions,
        folder_path=folder_path
    )

    cprint('---------- Done ----------', 'grey', 'on_green')


# ----------------------------------------------------------------------------------------------------
#                                                   Script
# ----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = Parser(argtype=ArgType.HPSearch)
    args = parser.parse_args()

    args = Args.preprocess.bayesian_opt(args)

    main(args)
