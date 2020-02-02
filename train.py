import os
from termcolor import cprint

from src.MidiGenerator import MidiGenerator
from src.NN.KerasNeuralNetwork import KerasNeuralNetwork
from src import Args
from src.Args import ArgType, Parser


def main(args):
    """
        Entry point
    """

    if args.pc:
        # args.data = 'lmd_matched_mini'
        data_path = os.path.join('../Dataset', args.data)
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'
    if args.mono:
        data_transformed_path += 'Mono'

    # -------------------- Create model --------------------
    midi_generator = MidiGenerator(name=args.name)
    # Choose GPU
    if not args.pc:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.debug:
        pass

    if args.no_eager:
        KerasNeuralNetwork.disable_eager_exection()

    if args.model_id != '':
        midi_generator.load_data(data_transformed_path=data_transformed_path)
        opt_param = dict(
            lr=args.lr,
            name=args.optimizer,
            decay_drop=args.decay_drop,
            epoch_drop=args.epochs_drop,
            decay=args.decay
        )
        model_options = dict(
            dropout_d=args.dropout_d,
            dropout_c=args.dropout_c,
            dropout_r=args.dropout_r,
            all_sequence=args.all_sequence,
            lstm_state=args.lstm_state,
            sampling=not args.no_sampling,
            kld=not args.no_kld,
            kld_annealing_start=args.kld_annealing_start,
            kld_annealing_stop=args.kld_annealing_stop,
            kld_sum=not args.no_kld_sum,
            sah=args.sah
        )
        loss_options = dict(
            loss_name=args.loss_name,
            l_scale=args.l_scale,
            l_rhythm=args.l_rhythm,
            l_scale_cost=args.l_scale_cost,
            l_rhythm_cost=args.l_rhythm_cost,
            take_all_step_rhythm=not args.no_all_step_rhythm
        )
        midi_generator.new_nn_model(model_id=args.model_id,
                                    opt_param=opt_param,
                                    work_on=args.work_on,
                                    model_options=model_options,
                                    loss_options=loss_options,
                                    predict_offset=args.predict_offset)
    elif args.load != '':
        midi_generator.recreate_model(args.load)

    # -------------------- Train --------------------
    midi_generator.train(epochs=args.epochs, batch=args.batch, noise=args.noise, validation=args.validation,
                         sequence_to_numpy=args.seq2np, fast_sequence=args.fast_seq, memory_sequence=args.memory_seq)

    # -------------------- Test --------------------
    if args.evaluate:
        midi_generator.evaluate()

    # -------------------- Test overfit --------------------
    if args.compare_generation:
        midi_generator.compare_generation(max_length=None,
                                          no_duration=args.no_duration,
                                          verbose=1)

    # -------------------- Generate --------------------
    if args.generate:
        midi_generator.generate_from_data(nb_seeds=4, save_images=True, no_duration=args.no_duration)

    # -------------------- Replicate --------------------
    if args.replicate:
        midi_generator.replicate_fom_data(save_images=True, no_duration=args.no_duration, noise=args.noise)

    # -------------------- Generate --------------------
    if args.generate_fill:
        midi_generator.generate_fill(no_duration=args.no_duration, verbose=1)

    if args.replicate_fill:
        midi_generator.replicate_fill(save_images=True, no_duration=args.no_duration, verbose=1, noise=args.noise)

    # -------------------- Debug batch generation --------------------
    if args.check_batch > -1:
        for i in range(len(midi_generator.sequence)):
            midi_generator.compare_test_predict_on_batch(i)

    # -------------------- Save the model --------------------
    midi_generator.save_model()

    cprint('---------- Done ----------', 'grey', 'on_green')


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    parser = Parser(argtype=ArgType.Train)
    args = parser.parse_args()

    args = Args.preprocess.train(args)
    main(args)
