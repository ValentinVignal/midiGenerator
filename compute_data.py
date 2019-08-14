import argparse
import os
import pickle
import numpy as np
from pathlib import Path
import progressbar
import shutil
from termcolor import colored

import src.midi.open as midi_open
import src.global_variables as g
import src.text.summary as summary



def all_midi_files(path, small_data):
    """

    :param path: the root path
    :param small_data: if we want to keep only a small amount of data
    :return: An array of all the path of all the .mid files in the directory
    """
    nb_small_data = 10
    fichiers = []
    if small_data:
        j = 0
        for root, dirs, files in os.walk(path):
            if j == nb_small_data:
                break
            for i in files:
                if j == nb_small_data:
                    break
                if i.endswith('.mid'):
                    fichiers.append(os.path.join(root, i))
                    j += 1
    else:
        for root, dirs, files in os.walk(path):
            for i in files:
                if i.endswith('.mid'):
                    fichiers.append(os.path.join(root, i))

    return fichiers


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_small',
                        help='The name of the data')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If data already exists, erase it and reconstruct it')
    parser.add_argument('-l', '--length', type=str, default='',
                        help='The length of the data')
    parser.add_argument('--notes_range', type=str, default='',
                        help='The length of the data')
    parser.add_argument('-i', '--instruments', type=str, default='Piano,Trombone',
                        help='The instruments considered (for space in name, put _ instead : Acoustic_Bass)')

    args = parser.parse_args()

    if args.pc:
        #args.data = 'lmd_matched_mini'
        data_path = os.path.join('../Dataset', args.data)
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'
    if args.force and os.path.exists(data_transformed_path):  # Delete the folder of the transformed data
        shutil.rmtree(data_transformed_path)
    if not os.path.exists(data_transformed_path):
        os.mkdir(data_transformed_path)

    if args.length == '':
        args.length = None
    else:
        args.length = int(args.length)

    if args.notes_range == '':
        args.notes_range = (0, 88)
    else:
        s = args.notes_range.split(':')
        args.notes_range = (int(s[0]), int(s[1]))

    # Instruments :
    args.instruments = list(map(lambda instrument: ' '.join(instrument.split('_')),
                                args.instruments.split(',')))
    #instruments = ['Piano', 'Trombone']

    all_dataset_p = os.path.join(data_transformed_path,
                                 'all_dataset.p')  # Pickle file with the informations of the data set
    dataset_p = os.path.join(data_transformed_path,
                             'dataset.p')  # Pickle file with the informations of the data set kept
    infos_dataset_p = os.path.join(data_transformed_path,
                                   'infos_dataset.p')  # pickle file with the informations of the dataset (smaller file)
    all_midi_paths = None
    #
    if os.path.exists(dataset_p):  # If we already know what are the correct files
        with open(dataset_p, 'rb') as dump_file:
            d = pickle.load(dump_file)
            all_midi_paths = d['midi']  # All the path for the files with no errors
    elif os.path.exists(all_dataset_p):  # If the path of all the files have already be saved
        with open(all_dataset_p, 'rb') as dump_file:
            d = pickle.load(dump_file)
            all_midi_paths_dataset = d[
                'midi']  # All the path for every files in the dataset (including the ones with errors)
    else:
        all_midi_paths_dataset = all_midi_files(data_path, args.pc)
        with open(all_dataset_p, 'wb') as dump_file:
            pickle.dump({
                'midi': all_midi_paths_dataset,
            }, dump_file)

    # From here either all_midi_path and all_midi_path_dataset is not None

    ##################################
    ##################################
    ##################################

    npy_path = os.path.join(data_transformed_path, 'npy')
    npy_pathlib = Path(npy_path)
    npy_pathlib.mkdir(parents=True, exist_ok=True)

    all_shapes = []

    # ----- Actually compute the datas -----
    if all_midi_paths is not None:
        # We already know what are the good files
        print('Compute the data in {0}'.format(data_path))
        matrix_of_all_midis = []

        # All midi have to be in same shape.
        bar = progressbar.ProgressBar(maxval=len(all_midi_paths),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        i = 0
        all_shapes_npy = []
        for single_midi_path in all_midi_paths:
            matrix_of_single_midi = midi_open.midi_to_matrix(single_midi_path, args.instruments,
                                                             length=args.length,
                                                             notes_range=args.notes_range)  # (nb_instruments, 88, nb_steps, 2)
            matrix_of_single_midi = np.transpose(matrix_of_single_midi, (2, 0, 1, 3))
            matrix_of_all_midis.append(matrix_of_single_midi)  # (length, nb_instruments, 128, 2)
            # print('shape of the matrix : {0}'.format(matrix_of_single_midi.shape))
            bar.update(i)
            i += 1
            all_shapes_npy.append(matrix_of_single_midi.shape)
            if i % g.nb_file_per_npy == 0:  # Save a npy file with 100 songs in it
                np.save(str(npy_pathlib / '{0}.npy'.format(int(i / g.nb_file_per_npy) - 1)), {
                    'list': matrix_of_all_midis,
                    'shapes': all_shapes_npy
                })
                matrix_of_all_midis = []
                all_shapes.append(all_shapes_npy)
                all_shapes_npy = []
        if i % g.nb_file_per_npy != 0:  # If some songs are missing
            np.save(str(npy_pathlib / '{0}.npy'.format(int(i / g.nb_file_per_npy))), {
                'list': matrix_of_all_midis,
                'shapes': all_shapes_npy
            })
            all_shapes.append(all_shapes_npy)
        bar.finish()
    else:
        print('Compute the data in {0}'.format(data_path))
        matrix_of_all_midis = []
        all_midi_paths = []

        # All midi have to be in same shape.
        bar = progressbar.ProgressBar(maxval=len(all_midi_paths_dataset),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        i_bar = 0
        i = 0
        all_shapes_npy = []
        for single_midi_path in all_midi_paths_dataset:
            matrix_of_single_midi = midi_open.midi_to_matrix(single_midi_path, args.instruments,
                                                             length=args.length,
                                                             notes_range=args.notes_range)  # (nb_instruments, 88, nb_steps, 2)
            if matrix_of_single_midi is not None:
                all_midi_paths.append(single_midi_path)
                matrix_of_single_midi = np.transpose(matrix_of_single_midi,
                                                     (2, 0, 1, 3))  # (length, nb_instruments, 88, 3)
                matrix_of_all_midis.append(matrix_of_single_midi)
                # print('shape of the matrix : {0}'.format(matrix_of_single_midi.shape))
                i += 1
                all_shapes_npy.append(matrix_of_single_midi.shape)
                if i % g.nb_file_per_npy == 0:  # Save 1 npy file with 100 songs in it
                    np.save(str(npy_pathlib / '{0}.npy'.format(int(i / g.nb_file_per_npy) - 1)), {
                        'list': matrix_of_all_midis,
                        'shapes': all_shapes_npy
                    })
                    all_shapes.append(all_shapes_npy)
                    all_shapes_npy = []
                    matrix_of_all_midis = []
            bar.update(i_bar)
            i_bar += 1
        if i % g.nb_file_per_npy != 0:  # If some songs are missing
            np.save(str(npy_pathlib / '{0}.npy'.format(int(i / g.nb_file_per_npy))), {
                'list': matrix_of_all_midis,
                'shapes': all_shapes_npy
            })
            all_shapes.append(all_shapes_npy)
        with open(dataset_p, 'wb') as dump_file:
            pickle.dump({
                'midi': all_midi_paths
            }, dump_file)
        bar.finish()
    # Now all_midi_paths is defined and we don't need all_midi_paths_dataset anymore

    nb_valid_files = len(all_midi_paths)

    with open(infos_dataset_p, 'wb') as dump_file:
        # Save the information of the data in a smaller file (without all the big array)
        pickle.dump({
            'nb_files': nb_valid_files,
            'instruments': args.instruments,
            'nb_instruments': len(args.instruments),
            'all_shapes': all_shapes,
            'input_size': all_shapes[0][0][2],  # The number of notes
            'notes_range': args.notes_range
        }, dump_file)

    summary.summarize_compute_data(data_transformed_path,
                                   **{
                                       'data_name': args.data,
                                       'nb_files': nb_valid_files,
                                       'nb_instruments':  len(args.instruments),
                                       'instruments': args.instruments,
                                       'input_size': all_shapes[0][0][2],
                                       'notes_range': args.notes_range
                                   })

    print('Number of songs :', colored('{0}'.format(nb_valid_files), 'blue'))


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
