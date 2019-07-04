import random
import numpy as np
import src.midi as midi
import music21
import bottleneck
import argparse
import os
from pathlib import Path

import src.NN.nn as nn


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset')
    parser.add_argument('--data', type=str, default='lmd_matched_mini', metavar='N',
                        help='The name of the data')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('--model', type=str, default='1',
                        help='The model of the Neural Network used for the interpolation')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')

    args = parser.parse_args()

    if args.pc:
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'
    if not os.path.exists(data_transformed_path):
        os.mkdir(data_transformed_path)

    midis_array_path = os.path.join(data_transformed_path, 'midis_array.npy')
    pl_midis_array_path = Path(midis_array_path)  # Use the library pathlib
    midis_array = np.load(midis_array_path)

    midis_array = np.reshape(midis_array, (-1, 128))

    max_len = 18
    input_param = {
        'nb_steps': 18,
        'input_size': 128
    }
    model = nn.create_model(input_param=input_param)

    print('Generation...')

    generated_midis_path = 'generated_midis'
    generated_midis_pathlib = Path(generated_midis_path)
    generated_midis_pathlib.mkdir(parents=True, exist_ok=True)

    start_index = random.randint(0, len(midis_array) - max_len - 1)

    for temperature in [0.7, 2.7]:
        print('------ temperature:', temperature)
        generated_midi = midis_array[start_index: start_index + max_len]
        for i in range(100):
            samples = generated_midi[i:]
            expanded_samples = np.expand_dims(samples, axis=0)
            preds = model.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = midi.sample(preds, temperature)

            midi_list = []
            midi_list.append(generated_midi)
            midi_list.append(next_array)
            generated_midi = np.vstack(midi_list)

        generated_midi_final = np.transpose(generated_midi, (1, 0))
        output_notes = midi.matrix_to_midi(generated_midi_final, random=1)
        midi_stream = music21.stream.Stream(output_notes)
        midi_file_name = (str(generated_midis_pathlib / 'lstm_out_{}.mid'.format(temperature)))
        midi_stream.write('midi', fp=midi_file_name)
        parsed = music21.converter.parse(midi_file_name)
        for part in parsed.parts:
            part.insert(0, music21.instrument.Piano())
        parsed.write('midi', fp=midi_file_name)

    # To see values

    z = -bottleneck.partition(-preds, 20)[:20]
    print(z)
    print('max:', np.max(preds))


    print('Done')

if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
