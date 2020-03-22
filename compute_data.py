import os
import pickle
import numpy as np
from epicpath import EPath
import progressbar
import shutil
from termcolor import colored, cprint

import src.Midi as midi
from src import GlobalVariables as g
import src.text.summary as summary
from src import Args
from src.Args import Parser, ArgType

os.system('echo Start compute data')


def check_args(args):
    """

    :param args:
    :return:
    """
    if args.pc:
        # args.data = 'lmd_matched_mini'
        data_path = os.path.join('../Dataset', args.data)
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'
    if not args.no_transpose:
        data_transformed_path += 'Transposed'
    if args.mono:
        data_transformed_path += 'Mono'

    if os.path.exists(data_transformed_path):  # Delete the folder of the transformed data
        shutil.rmtree(data_transformed_path)
    if not os.path.exists(data_transformed_path):
        os.mkdir(data_transformed_path)

    return args, EPath(data_path), EPath(data_transformed_path)


def main(args):
    """
        Entry point
    """
    args, data_path, data_transformed_path = check_args(args)

    # --------------------------------------------------

    # ----- All the paths -----
    dataset_p = data_transformed_path / 'dataset.p'  # Pickle file with the information of the data set kept
    infos_dataset_p = EPath(
        data_transformed_path) / 'infos_dataset.p'  # pickle file with the information of the dataset (smaller file)
    all_midi_paths_dataset = midi.open.all_midi_files(data_path.as_posix(), False)

    # --------------------------------------------------
    #               Compute dataset
    # --------------------------------------------------

    npy_path = data_transformed_path / 'npy'
    npy_path.mkdir(parents=True, exist_ok=True)

    all_shapes = []

    # ----- Actually compute the datas -----
    print('----- Compute the data in', colored(data_path, 'grey', 'on_white'), '-----')
    print('Number of files : ', colored(len(all_midi_paths_dataset), 'magenta'))
    bach_string = ''
    if args.bach:
        bach_string = colored('(Bach)', 'magenta')
    print('Instruments :', colored(args.instruments, 'magenta'), bach_string, '-- Notes range :',
          colored(args.notes_range, 'magenta'))

    matrix_of_all_midis = []
    all_midi_paths = []

    # All Midi have to be in same shape.
    bar = progressbar.ProgressBar(maxval=len(all_midi_paths_dataset),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                           progressbar.ETA()])
    bar.start()  # To see it working
    i = 0
    all_shapes_npy = []
    for index, single_midi_path in enumerate(all_midi_paths_dataset):
        bar.update(index)
        # ---------- Get the matrix ----------
        if args.bach:
            matrix_of_single_midi = midi.open.midi_to_matrix_bach(single_midi_path,
                                                                  length=args.length,
                                                                  notes_range=args.notes_range,
                                                                  transpose=not args.no_transpose
                                                                  )  # (nb_instruments, 88, nb_steps, 2)
        else:
            matrix_of_single_midi = midi.open.midi_to_matrix(single_midi_path, args.instruments,
                                                             length=args.length,
                                                             notes_range=args.notes_range,
                                                             transpose=not args.no_transpose
                                                             )  # (nb_instruments, 88, nb_steps, 2)
        if matrix_of_single_midi is None:  # It means an error happened
            continue

        if args.mono:
            matrix_of_single_midi = midi.open.to_mono_matrix(matrix_of_single_midi)
        matrix_of_single_midi = np.transpose(matrix_of_single_midi,
                                             (2, 0, 1, 3))  # (length, nb_instruments, 88, 2)

        # ---------- Add the matrix ----------
        all_midi_paths.append(single_midi_path)
        matrix_of_all_midis.append(matrix_of_single_midi)
        # print('shape of the matrix : {0}'.format(matrix_of_single_midi.shape))
        all_shapes_npy.append(matrix_of_single_midi.shape)

        i += 1
        # ---------- Save it ----------
        if i % g.midi.nb_files_per_npy == 0:  # Save 1 npy file with 100 songs in it
            np.save(str(npy_path / '{0}.npy'.format(int(i / g.midi.nb_files_per_npy) - 1)), {
                'list': matrix_of_all_midis,
                'shapes': all_shapes_npy
            })
            all_shapes.append(all_shapes_npy)
            all_shapes_npy = []
            matrix_of_all_midis = []

    # ---------- If we didn't save at the end ----------
    if len(all_shapes_npy) > 0:  # If some songs are missing
        np.save(str(npy_path / '{0}.npy'.format(int(i / g.midi.nb_files_per_npy))), {
            'list': matrix_of_all_midis,
            'shapes': all_shapes_npy
        })
        all_shapes.append(all_shapes_npy)

    # ---------- Save the path of all the midis ----------
    with open(dataset_p, 'wb') as dump_file:
        pickle.dump({
            'Midi': all_midi_paths
        }, dump_file)
    bar.finish()
    # Now all_midi_paths is defined and we don't need all_midi_paths_dataset anymore

    nb_valid_files = len(all_midi_paths)

    # ---------- Save all the information of the dataset ----------
    with open(infos_dataset_p, 'wb') as dump_file:
        # Save the information of the data in a smaller file (without all the big array)
        pickle.dump({
            'nb_files': nb_valid_files,
            'instruments': args.instruments,
            'nb_instruments': len(args.instruments),
            'all_shapes': all_shapes,
            'input_size': all_shapes[0][0][2],  # The number of notes
            'notes_range': args.notes_range,
            'mono': args.mono,
            'nb_files_per_npy': g.midi.nb_files_per_npy,
            'transposed': not args.no_transpose
        }, dump_file)

    summary.summarize(
        # Function params
        path=data_transformed_path,
        title=args.data,
        # Summary params
        nb_files=nb_valid_files,
        nb_instruments=len(args.instruments),
        instruments=args.instruments,
        input_size=all_shapes[0][0][2],
        notes_range=args.notes_range
    )

    print('Number of songs :', colored('{0}'.format(nb_valid_files), 'blue'))
    print(colored('---------- Done ----------', 'grey', 'on_green'))


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    parser = Parser(argtype=ArgType.ComputeData)
    args = parser.parse_args()
    args = Args.preprocess.compute_data(args)
    main(args)

