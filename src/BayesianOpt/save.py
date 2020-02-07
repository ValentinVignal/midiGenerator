from skopt.plots import plot_objective, plot_evaluations
from matplotlib import pyplot as plt
from pathlib import Path
from termcolor import colored

from src import Path as mPath
from src import BayesianOpt as BO


def get_folder_path(id=None, name=None):
    """

    :param name: The name of the future saved folder
    :type id: The id of an existing folder
    :return: the path to the folder to save the results
    """
    if id is None:
        # Then it has to be a new folder
        name_str = f'_{name}' if name is not None else ''
        return mPath.new.unique(Path('hp_search', f'bayesian_opt{name_str}'), mandatory_ext=True)
    else:
        id_list = id.split('-')
        id_str = '_' + '_'.join([str(s) for s in id_list])
        return Path('hp_search', f'bayesian_opt{id_str}')


def save_evaluations(search_result, folder_path):
    """

    :param search_result:
    :param folder_path:
    :return:
    """
    ax = plot_evaluations(result=search_result)
    folder_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(folder_path / 'evaluations.png')
    plt.close()
    print(f'Evaluation saved at {folder_path / "evaluations.png"}')


def save_objective(search_result, folder_path):
    """

    :param search_result:
    :param folder_path:
    :return:
    """
    ax = plot_objective(result=search_result)
    folder_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(folder_path / 'objective.png')
    plt.close()
    print(f'Objective saved at {folder_path / "objective.png"}')


def save_sorted_results(folder_path, sorted_scores, default_params_dict=None):
    """
    Save the list of the sorted scores and the correspondings parameters
    :param default_params_dict:
    :param folder_path:
    :param sorted_scores:
    :return:
    """
    # Text File
    text = '\t\tSorted scores and parameters:\n\n'
    # CSV File
    keys = sorted_scores[0][1].keys()
    text_csv = 'Accuracy;' + ';'.join(keys) + '\n'

    if default_params_dict is not None:
        # Text File
        text += f'Default params:\n{default_params_dict}\n\n'

    # Text File:
    for score, param_dict in sorted_scores:
        # Text File
        text += f'{-score:%}\t->\t{param_dict}\n'
        # CSV File
        text_csv += (f'{-score:%};' + ';'.join([str(param_dict[k]) for k in keys]) + '\n').replace('.', ',')

    if default_params_dict is not None:
        # CSV File
        text_csv += '\n\nDefault params:\n\n'
        for k in default_params_dict:
            text_csv += f'{k};{default_params_dict[k]}\n'

    # Text File
    with open(folder_path / 'sorted_scores.txt', 'w') as f:
        f.write(text)
    # CSV file
    with open(folder_path / 'sorted_scores.csv', 'w') as f:
        f.write(text_csv)
    print(f'Sorted scores saved at {folder_path / "sorted_scores.txt"} and {folder_path / "sorted_scores.csv"}')


def save_best_result(folder_path, best_accuracy, param_dict, default_params_dict=None):
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

    if default_params_dict is not None:
        text += '\nDefault params:\n\n'
        for k in default_params_dict:
            text += f'{k} : '
            if isinstance(default_params_dict[k], float):
                text += f'{default_params_dict[k]:.3e}\t({default_params_dict[k]})\n'
            else:
                text += f'{default_params_dict[k]}\n'

    with open(folder_path / 'best_params.txt', 'w') as f:
        f.write(text)
    print(f'Best result saved in {folder_path / "best_params.txt"}')


def save_search_result(search_result, dimensions, folder_path):
    """

    :param folder_path:
    :param search_result:
    :param dimensions:
    :return:
    """
    best_param_dict = dimensions.point_to_dict(search_result.x)
    best_accuracy = - search_result.fun

    # ---------- Print the best results ----------
    # Print acc
    print('Best Result:', colored(f'{best_accuracy}', 'green'))
    s = ''
    # Print params
    for k in best_param_dict:
        if isinstance(best_param_dict[k], float):
            s += f'{k}:' + colored(f'{best_param_dict[k]:.1e}', 'magenta') + ' - '
        else:
            s += f'{k}:' + colored(f'{best_param_dict[k]}', 'magenta') + ' - '
    print(s)
    # Print default params
    s = 'With the default parameters:\n'
    for k in dimensions.default_params_dict:
        if isinstance(dimensions.default_params_dict[k], float):
            s += f'{k}:' + colored(f'{dimensions.default_params_dict[k]:.1e}', 'magenta') + ' - '
        else:
            s += f'{k}:' + colored(f'{dimensions.default_params_dict[k]}', 'magenta') + ' - '
    print(s)

    sorted_scores = sorted(
        zip(
            search_result.func_vals,
            [dimensions.point_to_dict(x_) for x_ in search_result.x_iters]
        ),
        key=lambda x: x[0]  # To sort only with the precision not the dictionaries
    )

    # ------------------------------
    #           Save it
    # ------------------------------


    # ---------- Text ----------

    # Save the best result hyper parameters in a .txt file
    BO.save.save_best_result(
        folder_path=folder_path,
        best_accuracy=best_accuracy,
        param_dict=best_param_dict,
        default_params_dict=dimensions.default_params_dict
    )
    # Save all the sorted results
    BO.save.save_sorted_results(
        folder_path=folder_path,
        sorted_scores=sorted_scores,
        default_params_dict=dimensions.default_params_dict
    )

    # ---------- Images ----------

    BO.save.save_objective(
        search_result=search_result,
        folder_path=folder_path,
    )

    BO.save.save_evaluations(
        search_result=search_result,
        folder_path=folder_path
    )

    print('Results saved in', colored(folder_path.as_posix(), 'green'))
