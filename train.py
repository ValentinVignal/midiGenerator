import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import json
from math import ceil


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
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batch to wait before logging training status')
    parser.add_argument('--model', type=str, default='1',
                        help='The model of the Neural Network used for the interpolation')
    parser.add_argument('--batch', type=int, default=1,
                        help='The number of the batchs')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')

    args = parser.parse_args()

    if args.pc:
        data_path = '../Dataset/lmd_matched'
    else:
        data_path = '../../../../../../storage1/valentin/lmd_matched'


    data_p = os.path.join(data_path, 'data.p')      # Pickle file with the informations of the data set
    if os.path.exists(data_p):
        with open(data_p, 'rb') as dump_file:
            d = pickle.load(dump_file)
            data_midi = d['midi']
    else:
        data_midi = allMidiFiles(data_path, args.pc)
        with open(data_p, 'wb') as dump_file:
            pickle.dump({
                'midi': data_midi
            }, dump_file)

    print(len(data_midi))







if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
