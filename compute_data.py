import argparse
import os
import pickle
import numpy as np
from pathlib import Path
import progressbar
import shutil
from termcolor import colored, cprint

import src.midi.open as midi_open
import src.global_variables as g
import src.text.summary as summary
import src.midi.instruments as midi_inst


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

    if os.path.exists(data_transformed_path):  # Delete the folder of the transformed data
        shutil.rmtree(data_transformed_path)
    if not os.path.exists(data_transformed_path):
        os.mkdir(data_transformed_path)

    if args.length == '':
        args.length = None
    else:
        args.length = int(args.length)

    s = args.notes_range.split(':')
    args.notes_range = (int(s[0]), int(s[1]))

    # Instruments :
    args.instruments = list(map(lambda instrument: ' '.join(instrument.split('_')),
                                args.instruments.split(',')))
    if args.bach:
        args.instruments = midi_inst.bach_instruments

    return args, Path(data_path), Path(data_transformed_path)


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, default='lmd_matched_small',
                        help='The name of the data')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('-l', '--length', type=str, default='',
                        help='The length of the data')
    parser.add_argument('--notes-range', type=str, default='0:88',
                        help='The length of the data')
    parser.add_argument('-i', '--instruments', type=str, default='Piano,Trombone',
                        help='The instruments considered (for space in name, put _ instead : Acoustic_Bass)')
    parser.add_argument('--bach', action='store_true', default=False,
                        help='To compute the bach data')
    parser.add_argument('--mono', action='store_true', default=False,
                        help='To compute the data where there is only one note at the same time')

    args = parser.parse_args()

    args, data_path, data_transformed_path = check_args(args)

    # --------------------------------------------------

    # ----- All the paths -----
    dataset_p = data_transformed_path / 'dataset.p'  # Pickle file with the information of the data set kept
    infos_dataset_p = Path(
        data_transformed_path) / 'infos_dataset.p'  # pickle file with the information of the dataset (smaller file)
    all_midi_paths_dataset = midi_open.all_midi_files(data_path.as_posix(), False)

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

    # All midi have to be in same shape.
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
            matrix_of_single_midi = midi_open.midi_to_matrix_bach(single_midi_path,
                                                                  length=args.length,
                                                                  notes_range=args.notes_range
                                                                  )  # (nb_instruments, 88, nb_steps, 2)
        else:
            matrix_of_single_midi = midi_open.midi_to_matrix(single_midi_path, args.instruments,
                                                             length=args.length,
                                                             notes_range=args.notes_range
                                                             )  # (nb_instruments, 88, nb_steps, 2)
        if matrix_of_single_midi is None:  # It means an error happened
            continue

        if args.mono:
            matrix_of_single_midi = midi_open.to_one_note_matrix(matrix_of_single_midi)
        matrix_of_single_midi = np.transpose(matrix_of_single_midi,
                                             (2, 0, 1, 3))  # (length, nb_instruments, 88, 2)

        # ---------- Add the matrix ----------
        all_midi_paths.append(single_midi_path)
        matrix_of_all_midis.append(matrix_of_single_midi)
        # print('shape of the matrix : {0}'.format(matrix_of_single_midi.shape))
        all_shapes_npy.append(matrix_of_single_midi.shape)

        i += 1
        # ---------- Save it ----------
        if i % g.nb_file_per_npy == 0:  # Save 1 npy file with 100 songs in it
            np.save(str(npy_path / '{0}.npy'.format(int(i / g.nb_file_per_npy) - 1)), {
                'list': matrix_of_all_midis,
                'shapes': all_shapes_npy
            })
            all_shapes.append(all_shapes_npy)
            all_shapes_npy = []
            matrix_of_all_midis = []

    # ---------- If we didn't save at the end ----------
    if len(all_shapes_npy) > 0:  # If some songs are missing
        np.save(str(npy_path / '{0}.npy'.format(int(i / g.nb_file_per_npy))), {
            'list': matrix_of_all_midis,
            'shapes': all_shapes_npy
        })
        all_shapes.append(all_shapes_npy)

    # ---------- Save the path of all the midis ----------
    with open(dataset_p, 'wb') as dump_file:
        pickle.dump({
            'midi': all_midi_paths
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
            'mono': args.mono
        }, dump_file)

    summary.summarize_compute_data(data_transformed_path,
                                   **{
                                       'data_name': args.data,
                                       'nb_files': nb_valid_files,
                                       'nb_instruments': len(args.instruments),
                                       'instruments': args.instruments,
                                       'input_size': all_shapes[0][0][2],
                                       'notes_range': args.notes_range
                                   })

    print('Number of songs :', colored('{0}'.format(nb_valid_files), 'blue'))
    print(colored('---------- Done ----------', 'grey', 'on_green'))


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
