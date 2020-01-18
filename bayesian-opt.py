import os
from termcolor import cprint, colored
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import gc

from src import skopt_
from src.skopt_.space import Real, Categorical
from src.skopt_.utils import use_named_args
from src.skopt_ import gp_minimize

K = tf.keras.backend

from src.MidiGenerator import MidiGenerator
from src.NN.Callbacks import LossHistory
import src.global_variables as g
from src import Args
from src.Args import ArgType, Parser


def create_list(string):
    """

    :param string:
    :return:
    """
    string_list = string.split(':')
    if len(string_list) == 1:
        return [float(string_list[0])]
    else:
        return list(
            np.arange(float(string_list[0]), float(string_list[1]) + float(string_list[2]), float(string_list[2])))


def get_tuple(string, t=float, separator=':'):
    """
    Get the tuple from the string given by the user
    :param separator: separator of the value in the string
    :param t: float, int or other type (function)
    :param string: the string given by the user
    :return:
    """
    return tuple(map(t, string.split(separator)))


def str2bool(string):
    return string == 'True'


def create_dimensions(args):
    """

    :param args:
    :return:
    """
    dimensions = []
    default_dim = []

    def add_Real(mtuple, name, prior='uniform'):
        if len(mtuple) == 1:
            dimensions.append(Categorical([mtuple[0]], name=name))
            default_dim.append(mtuple[0])
        elif len(mtuple) == 2:
            dimensions.append(Real(low=min(mtuple), high=max(mtuple), name=name, prior=prior))
            default_dim.append(sum(mtuple) / len(mtuple))

    def add_Categorical(m_tuple, name):
        dimensions.append(Categorical(list(m_tuple), name=name))
        default_dim.append(m_tuple[0])

    # lr
    lr_tuple = get_tuple(args.lr)
    lr_tuple = tuple(map(lambda x: 10 ** (-x), lr_tuple))
    add_Real(lr_tuple, name='lr', prior='log-uniform')
    # optimizer
    opt_tuple = get_tuple(args.optimizer, t=str, separator=',')
    add_Categorical(opt_tuple, name='optimizer')
    # decay
    decay_tuple = get_tuple(args.decay)
    add_Real(decay_tuple, name='decay')
    # dropout
    dropout_tuple = get_tuple(args.dropout)
    add_Real(dropout_tuple, name='dropout')
    # sampling
    sampling_tuple = get_tuple(args.no_sampling, t=lambda x: not str2bool(x), separator=',')
    add_Categorical(sampling_tuple, name='sampling')
    # kld
    kld_tuple = get_tuple(args.no_kld, t=lambda x: not str2bool(x), separator=',')
    add_Categorical(kld_tuple, name='kld')
    # all sequence
    all_sequence_tuple = get_tuple(args.all_sequence, t=str2bool, separator=',')
    add_Categorical(all_sequence_tuple, name='all_sequence')
    # model name
    model_name_tuple = get_tuple(args.model_name, t=str, separator=',')
    add_Categorical(model_name_tuple, 'model_name')
    # model param
    model_param_tuple = get_tuple(args.model_param, t=str, separator=',')
    add_Categorical(model_param_tuple, 'model_param')
    # nb steps
    nb_steps_tuple = get_tuple(args.nb_steps, t=int, separator=',')
    add_Categorical(nb_steps_tuple, 'nb_steps')
    # kld annealing start
    kld_annealing_start_tuple = get_tuple(args.kld_annealing_start)
    add_Real(kld_annealing_start_tuple, name='kld_annealing_start')
    # kld annealing stop
    kld_annealing_stop_tuple = get_tuple(args.kld_annealing_stop)
    add_Real(kld_annealing_stop_tuple, name='kld_annealing_stop')
    # kld sum
    kld_sum_tuple = get_tuple(args.no_kld_sum, t=lambda x: not str2bool(x), separator=',')
    add_Categorical(kld_tuple, name='kld')

    return dimensions, default_dim


def get_history_acc(history):
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


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------


def main(args):
    """
        Entry point
    """
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

    dimensions, default_dim = create_dimensions(args)

    global best_accuracy
    best_accuracy = 0
    global iteration
    iteration = 0

    def create_model(lr, optimizer, decay, dropout, sampling, kld, all_sequence, model_name, model_param, nb_steps,
                     kld_annealing_start, kld_annealing_stop, kld_sum):
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
            dropout=dropout,
            all_sequence=all_sequence,
            lstm_state=args.lstm_state,
            sampling=sampling,
            kld=kld,
            kld_annealing_start=kld_annealing_start,
            kld_annealing_stop=kld_annealing_stop,
            kld_sum=kld_sum
        )

        midi_generator.new_nn_model(
            model_id=model_id,
            opt_param=opt_params,
            work_on=args.work_on,
            type_loss=args.type_loss,
            model_options=model_options,
            print_model=False
        )
        return midi_generator

    @use_named_args(dimensions=dimensions)
    def fitness(lr, optimizer, decay, dropout, sampling, kld, all_sequence, model_name, model_param, nb_steps,
                kld_annealing_start, kld_annealing_stop, kld_sum):
        global iteration
        iteration += 1

        s = 'Iteration ' + colored(f'{iteration}/{args.n_calls}', 'yellow')
        model_id = f'{model_name},{model_param},{nb_steps}'
        s += str_hp_to_print('model', model_id, first_printed=False)
        s += str_hp_to_print('lr', lr, exp_format=True)
        s += str_hp_to_print('opt', optimizer)
        s += str_hp_to_print('decay', decay, exp_format=True)
        s += str_hp_to_print('dropout', dropout, exp_format=True)
        s += str_hp_to_print('sampling', sampling)
        s += str_hp_to_print('kld', kld)
        s += str_hp_to_print('all_sequence', all_sequence)
        s += str_hp_to_print('kld_annealing_start', kld_annealing_start, exp_format=True)
        s += str_hp_to_print('kld_annealing_stop', kld_annealing_stop, exp_format=True)
        s += str_hp_to_print('kld_sum', kld_sum)

        print(s)

        midi_generator = create_model(
            lr=lr,
            optimizer=optimizer,
            decay=decay,
            dropout=dropout,
            sampling=sampling,
            kld=kld,
            all_sequence=all_sequence,
            model_name=model_name,
            model_param=model_param,
            nb_steps=nb_steps,
            kld_annealing_start=kld_annealing_start,
            kld_annealing_stop=kld_annealing_stop,
            kld_sum=kld_sum
        )
        history = midi_generator.train(epochs=args.epochs, batch=args.batch, callbacks=[], verbose=1,
                                       validation=args.validation)
        accuracy = get_history_acc(history)

        global best_accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        print(f'Accuracy:',
              colored(f'{accuracy:.2%}', 'cyan'),
              '- Best Accuracy for now:',
              colored(f'{best_accuracy:.2%}', 'white', 'on_blue'))

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

        return -accuracy

    # ------------------------------
    #           Run it
    # ------------------------------

    search_result = gp_minimize(
        func=fitness,
        dimensions=dimensions,
        acq_func='EI',
        n_calls=args.n_calls,
        x0=default_dim
    )
    space = search_result.space

    best_param_dict = space.point_to_dict(search_result.x)
    best_accuracy = - search_result.fun

    # ----- Print the best results -----
    print('Best Result:', colored(f'{best_accuracy}', 'green'))
    s = ''
    for k in best_param_dict:
        if isinstance(best_param_dict[k], float):
            s += f'{k}:' + colored(f'{best_param_dict[k]:.1e}', 'magenta') + ' - '
        else:
            s += f'{k}:' + colored(f'{best_param_dict[k]}', 'magenta') + ' - '

    print(s)

    sorted_scores = sorted(
        zip(
            search_result.func_vals,
            [space.point_to_dict(x_) for x_ in search_result.x_iters]
        ),
        key=lambda x: x[0]      # To sort only with the precision not the dictionaries
    )

    # ------------------------------
    #           Save it
    # ------------------------------

    # ---------- Text ----------

    folder_path = get_folder_path()
    folder_path.mkdir(exist_ok=True, parents=True)

    save_best_result(
        folder_path=folder_path,
        best_accuracy=best_accuracy,
        param_dict=best_param_dict
    )
    save_sorted_results(
        folder_path=folder_path,
        sorted_scores=sorted_scores
    )

    # ---------- Images ----------

    save_histogram(
        search_result=search_result,
        folder_path=folder_path
    )

    save_objective_2D(
        search_result=search_result,
        folder_path=folder_path
    )

    save_objective(
        search_result=search_result,
        folder_path=folder_path
    )

    print('Results saved in', colored(folder_path.as_posix(), 'green'))

    cprint('---------- Done ----------', 'grey', 'on_green')


def save_histogram(search_result, folder_path):
    """

    :param search_result:
    :param folder_path:
    :return:
    """
    histogram_folder_path = folder_path / 'histogram'
    histogram_folder_path.mkdir(exist_ok=True, parents=True)
    for dim_name in search_result.space.dimension_names:
        fig, ax = skopt_.plots.plot_histogram(
            result=search_result,
            dimension_name=dim_name
        )
        fig.savefig(histogram_folder_path / f'{dim_name}.png')
        plt.close(fig)


def save_objective(search_result, folder_path):
    """

    :param search_result:
    :param folder_path:
    :return:
    """
    dimension_names = [
        name for name in search_result.space.dimension_names if not isinstance(search_result.space[name], Categorical)
    ]
    fig, ax = skopt_.plots.plot_objective(result=search_result, dimension_names=dimension_names)
    folder_path.mkdir(exist_ok=True, parents=True)
    fig.savefig(folder_path / 'objective.png')
    plt.close(fig)


def save_objective_2D(search_result, folder_path):
    """

    :param folder_path:
    :param search_result:
    :return:
    """
    dimensions_names = search_result.space.dimension_names
    objective_folder_path = folder_path / 'objective_2D'
    objective_folder_path.mkdir(exist_ok=True, parents=True)
    for i in range(len(dimensions_names)):
        for j in range(i + 1, len(dimensions_names)):
            name1, name2 = dimensions_names[i], dimensions_names[j]
            if not any(isinstance(search_result.space[name], Categorical) for name in [name1, name2]):
                fig, ax = skopt_.plots.plot_objective_2D(
                    result=search_result,
                    dimension_name1=name1,
                    dimension_name2=name2,
                    levels=50
                )
                fig.savefig(objective_folder_path / f'{name1}-{name2}.png')
                plt.close(fig)


def save_sorted_results(folder_path, sorted_scores):
    """
    Save the list of the sorted scores and the correspondings parameters
    :param folder_path:
    :param sorted_scores:
    :return:
    """
    text = '\t\tSorted scores and parameters:\n\n'
    for score, param_dict in sorted_scores:
        text += f'{score:%}\t->\t{param_dict}\n'
        """
        for k in range(len(dim_names)):
            text += f'{dim_names[i]}: {param_list[i]} - '
        text += '\n'
        """
    with open(folder_path / 'sorted_scores.txt', 'w') as f:
        f.write(text)


def save_best_result(folder_path, best_accuracy, param_dict):
    """
    Save the best results and the parameters in a folder path
    :param best_accuracy:
    :param param_dict:
    :param folder_path:
    :return:
    """
    text = f'\t\tAccuracy: {best_accuracy:%}\n\n'
    text += 'Params:\n'
    for k in param_dict:
        text += f'{k} : '
        if isinstance(param_dict[k], float):
            text += f'{param_dict[k]:.3e}\t({param_dict[k]})\n'
        else:
            text += f'{param_dict[k]}\n'
    with open(folder_path / 'best_params.txt', 'w') as f:
        f.write(text)


def get_folder_path():
    """

    :return: the path to the folder to save the results
    """
    folder_path = Path('hp_search')
    i = 0
    while (folder_path / f'bayesian_opt_{i}').exists():
        i += 1
    folder_path = folder_path / f'bayesian_opt_{i}'
    return folder_path


if __name__ == '__main__':
    parser = Parser(argtype=ArgType.HPSearch)
    args = parser.parse_args()

    args = Args.preprocess.bayesian_opt(args)

    main(args)
