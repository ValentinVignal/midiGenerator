import argparse
import os
import pickle
import numpy as np
from pathlib import Path
import progressbar

import src.midi as midi

def allMidiFiles(path, small_data):
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

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset')
    parser.add_argument('--data', type=str, default='lmd_matched_mini', metavar='N',
                        help='The name of the data')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')

    args = parser.parse_args()

    if args.pc:
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'
    if not os.path.exists(data_transformed_path):
        os.mkdir(data_transformed_path)

    all_dataset_p = os.path.join(data_transformed_path, 'all_dataset.p')      # Pickle file with the informations of the data set
    data_p = os.path.join(data_transformed_path, 'data.p')        # Pickle file with the informations of the data set kept
    all_midi_paths = None
    if os.path.exists(data_p):
        with open(data_p, 'rb') as dump_file:
            d = pickle.load(dump_file)
            all_midi_paths = d['midi']      # All the path for the files with no errors
    elif os.path.exists(all_dataset_p):
        with open(all_dataset_p, 'rb') as dump_file:
            d = pickle.load(dump_file)
            all_midi_paths_dataset = d['midi']      # All the path for every files in the dataset (including the ones with errors)
    else:
        all_midi_paths_dataset = allMidiFiles(data_path, args.pc)
        with open(all_dataset_p, 'wb') as dump_file:
            pickle.dump({
                'midi': all_midi_paths_dataset
            }, dump_file)

    ##################################
    ##################################
    ##################################

    midis_array_path = os.path.join(data_transformed_path, 'midis_array.npy')
    midis_array_pathlib = Path(midis_array_path)

    if midis_array_pathlib.is_file():
        midis_array = np.load(midis_array_path)
    elif all_midi_paths is not None:
        print('Compute the data in {0}'.format(data_path))
        matrix_of_all_midis = []

        # All midi have to be in same shape.
        bar = progressbar.ProgressBar(maxval=len(all_midi_paths), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
        bar.start()     # To see it working
        i = 0
        for single_midi_path in all_midi_paths:
            matrix_of_single_midi = midi.midi_to_matrix(single_midi_path, length=250)
            if (matrix_of_single_midi is not None):
                matrix_of_all_midis.append(matrix_of_single_midi)
                # print('shape of the matrix : {0}'.format(matrix_of_single_midi.shape))
            bar.update(i)
            i += 1
        midis_array = np.asarray(matrix_of_all_midis)
        midis_array = np.transpose(midis_array, (0, 2, 1))
        np.save(midis_array_path, midis_array)
        bar.finish()
    else:
        print('Compute the data in {0}'.format(data_path))
        matrix_of_all_midis = []
        all_midi_paths = []

        # All midi have to be in same shape.
        bar = progressbar.ProgressBar(maxval=len(all_midi_paths_dataset), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
        bar.start()     # To see it working
        i = 0
        for single_midi_path in all_midi_paths_dataset:
            matrix_of_single_midi = midi.midi_to_matrix(single_midi_path, length=250)
            if matrix_of_single_midi is not None:
                all_midi_paths.append(single_midi_path)
                matrix_of_all_midis.append(matrix_of_single_midi)
                # print('shape of the matrix : {0}'.format(matrix_of_single_midi.shape))
            bar.update(i)
            i += 1
        with open(data_p, 'wb') as dump_file:
            pickle.dump({
                'midi': all_midi_paths
            }, dump_file)
        midis_array = np.asarray(matrix_of_all_midis)
        midis_array = np.transpose(midis_array, (0, 2, 1))
        np.save(midis_array_path, midis_array)
        bar.finish()
    # Now all_midi_paths is defined and we don't need all_midi_paths_dataset anymore

    midis_array = np.reshape(midis_array, (-1, 128))

    print('midis_array : {0}'.format(midis_array.shape))


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
