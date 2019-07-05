import argparse
import os

from src.NN.MyModel import MyModel




def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset')
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_mini', metavar='N',
                        help='The name of the data')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('-b', '--batch', type=int, default=1,
                        help='The number of the batchs')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    #parser.add_argument('--log-interval', type=int, default=5, metavar='N',
    #                    help='how many batch to wait before logging training status')
    parser.add_argument('-n', '--name', type=str, default='default_name',
                        help='how many batch to wait before logging training status')
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument('-m', '--model', type=str, default='',
                            help='The model of the Neural Network used for the interpolation')
    load_group.add_argument('-l', '--load', type=str, default='',
                            help='The name of the trained model to load')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')

    args = parser.parse_args()

    if args.pc:
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    my_model = MyModel()
    my_model.load_data(data_transformed_path=data_transformed_path)

    input_param = {
        'nb_steps': 18,
        'input_size': 128
    }

    #my_model.new_nn_model(input_param=input_param)
    #my_model.load_weights('default_name--0-1')
    my_model.load_model('default_name--2-0')
    my_model.train(epochs=args.epochs, batch=args.batch, verbose=1, shuffle=True)
    my_model.save_model()
    my_model.print_weights()


    #################################
    ####################################### Generation
    #######################################

    """
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

    model.save_weights(str(saved_models_pathlib / 'my_model_weights.h5'))
    model.save(str(saved_models_pathlib / 'my_model.h5'))

    print('Done')
    """


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
