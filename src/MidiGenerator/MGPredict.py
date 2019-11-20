import random
from termcolor import colored, cprint
import numpy as np
import progressbar

from src.NN import Sequences
import src.Midi as midi
import src.image.pianoroll as pianoroll
import src.global_variables as g
import src.text.summary as summary


class MGPredict:
    """

    """
    def generate_fom_data(self, nb_seeds=10, new_data_path=None, length=None, new_save_path=None, save_images=False,
                          no_duration=False, verbose=1):
        """
        Generate Midi file from the seed and the trained model
        :param nb_seeds: number of seeds for the generation
        :param new_data_path: The path of the seed
        :param length: Length of th generation
        :param new_save_path:
        :param save_images: To save the pianoroll of the generation (.jpg images)
        :param no_duration: if True : all notes will be the shortest length possible
        :param verbose: Level of verbose
        :return:
        """
        nb_steps = int(self.model_id.split(',')[2])

        # ---------- Verify the inputs ----------

        # ----- Create the seed -----
        need_new_sequence = False
        if (new_data_path is not None) and (new_data_path != self.data_transformed_pathlib.as_posix()):
            self.load_data(new_data_path)
            need_new_sequence = True
        if self.data_transformed_pathlib is None:
            raise Exception('Some data need to be loaded before generating')
        if self.my_sequence is None:
            need_new_sequence = True
        if need_new_sequence:
            self.my_sequence = Sequences.AllInstSequence(
                path=str(self.data_transformed_pathlib),
                nb_steps=nb_steps,
                batch_size=1,
                work_on=self.work_on)
        else:
            self.my_sequence.change_batch_size(1)

        seeds_indexes = random.sample(range(len(self.my_sequence)), nb_seeds)

        step_length = g.work_on2nb(self.work_on)
        # -- Length --
        length = length if length is not None else 200
        # -- For save Midi path --
        if type(new_save_path) is str or (
                type(new_save_path) is bool and new_save_path) or (
                new_save_path is None and self.save_midis_pathlib is None):
            self.get_new_save_midis_path(path=new_save_path)
        # --- Done Verifying the inputs ---

        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)
        cprint('Start generating ...', 'blue')
        for s in range(nb_seeds):
            cprint('Generation {0}/{1}'.format(s + 1, nb_seeds), 'blue')
            generated = np.array(
                self.my_sequence[seeds_indexes[s]][0])  # (nb_instruments, 1, nb_steps, step_size, inputs_size, 2)
            bar = progressbar.ProgressBar(maxval=length,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                                   progressbar.ETA()])
            bar.start()  # To see it working
            for l in range(length):
                samples = generated[:, :, l:]  # (nb_instruments, 1, nb_steps, length, 88, 2)   # 1 = batch
                # expanded_samples = np.expand_dims(samples, axis=0)
                preds = self.keras_nn.generate(
                    input=list(samples))  # (nb_instruments, batch=1 , nb_steps=1, length, 88, 2)
                preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, 1, step_size, input_size, 2)
                if len(preds.shape) == 4:  # Only one instrument : output of nn not a list
                    preds = preds[np.newaxis]
                next_array = midi.create.normalize_activation(preds)  # Normalize the activation part
                generated = np.concatenate((generated, next_array), axis=2)  # (nb_instruments, nb_steps, length, 88, 2)

                bar.update(l + 1)
            bar.finish()

            generated_midi_final = np.reshape(generated, (
                generated.shape[0], generated.shape[2] * generated.shape[3], generated.shape[4],
                generated.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
            generated_midi_final = np.transpose(generated_midi_final,
                                                (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
            output_notes_list = midi.create.matrix_to_midi(generated_midi_final, instruments=self.instruments,
                                                           notes_range=self.notes_range, no_duration=no_duration,
                                                           mono=self.mono)

            # --- find the name for the midi_file ---
            i = 0
            m_str = "lstm_out_({0}).mid".format(i)
            while (self.save_midis_pathlib / m_str).exists():
                i += 1
                m_str = "lstm_out_({0}).mid".format(i)
            path_to_save = str(self.save_midis_pathlib / m_str)
            path_to_save_img = str(self.save_midis_pathlib / 'lstm_out_({0}).jpg'.format(i))

            midi.create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final,
                                           notes_list=output_notes_list, verbose=verbose)

            # Saving the Midi file
            midi.create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments,
                                  path=path_to_save, )
            if save_images:
                pianoroll.save_pianoroll(array=generated_midi_final,
                                         path=path_to_save_img,
                                         seed_length=nb_steps * step_length,
                                         instruments=self.instruments,
                                         mono=self.mono)

        if self.batch is not None:
            self.my_sequence.change_batch_size(self.batch)

        summary.summarize_generation(str(self.save_midis_pathlib), **{
            'full_name': self.full_name,
            'epochs': self.total_epochs,
            'input_param': self.input_param,
            'instruments': self.instruments,
            'notes_range': self.notes_range
        })

        cprint('Done Generating', 'green')

    def fill_instruments(self, max_length=None, no_duration=False, verbose=1):
        """

        :param max_length:
        :param no_duration:
        :param verbose:
        :return:
        """
        # ----- Parameters -----
        max_length = 300 / g.work_on2nb(self.work_on) if max_length is None else max_length

        # ----- Variables -----
        if self.data_transformed_pathlib is None:
            raise Exception('Some data need to be loaded before comparing the generation')
        sequence = Sequences.KerasSequence(
            path=self.data_transformed_pathlib,
            nb_steps=self.nb_steps,
            batch_size=1,
            work_on=self.work_on
        )  # Return array instead of list (for instruments)
        max_length = int(min(max_length, len(sequence)))
        nb_instruments = sequence.nb_instruments
        # ----- Seeds -----
        truth = sequence[0][0]
        filled_list = [np.copy(truth) for inst in range(nb_instruments)]
        mask = np.ones((nb_instruments, nb_instruments, self.nb_steps))
        for inst in range(nb_instruments):
            filled_list[inst][inst] = 0
            mask[inst, inst] = 0

        # ----- Generation -----
        bar = progressbar.ProgressBar(maxval=max_length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(max_length):
            s_input, s_output = sequence[l]
            to_fill_list = [np.copy(s_input) for inst in range(nb_instruments)]
            for inst in range(nb_instruments):
                to_fill_list[inst][inst] = 0
            nn_input = np.concatenate(
                tuple(to_fill_list),
                axis=1
            )  # (nb_instruments, batch=nb_instruments, nb_steps, step_size, input_size, channels)
            preds = self.keras_nn.generate(input=list(nn_input) + [mask])

            preds = np.asarray(preds).astype(
                'float64')  # (nb_instruments, bath=nb_instruments, nb_steps=1, step_size, input_size, channels)
            if len(preds.shape) == 5:  # Only one instrument : output of nn not a list
                preds = np.expand_dims(preds, axis=0)
            if len(s_output.shape) == 5:  # Only one instrument : output of nn not a list
                s_output = np.expand_dims(s_output)
            truth = np.concatenate((truth, s_output), axis=2)
            for inst in range(nb_instruments):
                p = np.copy(s_output)
                p[inst] = np.take(preds, axis=1, indices=[inst])[inst]
                filled_list[inst] = np.concatenate(
                    (filled_list[inst], p),
                    axis=2)  # (nb_instruments, batch=1, nb_steps, step_size, input_size, channels)
            bar.update(l + 1)
        bar.finish()

        # -------------------- Compute notes list --------------------
        # ----- Reshape -----
        truth = np.reshape(truth, (truth.shape[0], truth.shape[2] * truth.shape[3],
                                   *truth.shape[
                                    4:]))  # (nb_instruments, nb_steps * step_size, input_size, channels)
        truth = np.transpose(truth, axes=(0, 2, 1, 3))  # (nb_instruments, input_size, length, channels)
        for inst in range(nb_instruments):
            s = filled_list[inst].shape
            filled_list[inst] = np.reshape(filled_list[inst], (
                s[0], s[2] * s[3], *s[4:]))  # (nb_instruments, nb_steps * step_size, input_size, channels)
            filled_list[inst] = np.transpose(filled_list[inst],
                                             axes=(0, 2, 1, 3))  # (nb_instruments, input_size, length, channels)
        # ----- Notes -----
        output_notes_truth = midi.create.matrix_to_midi(truth,
                                                        instruments=self.instruments,
                                                        notes_range=self.notes_range, no_duration=no_duration,
                                                        mono=self.mono)
        output_notes_inst = [
            midi.create.matrix_to_midi(filled_list[inst], instruments=self.instruments,
                                       notes_range=self.notes_range,
                                       no_duration=no_duration, mono=self.mono)
            for inst in range(nb_instruments)
        ]

        # ---------- find the name for the midi_file ----------
        self.get_new_save_midis_path()
        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)

        # -------------------- Save Results --------------------
        # Truth
        midi.create.print_informations(nb_steps=self.nb_steps * self.step_length,
                                       matrix=truth, notes_list=output_notes_truth, verbose=verbose)
        # TODO: Accuracy
        midi.create.save_midi(output_notes_list=output_notes_truth, instruments=self.instruments,
                              path=self.save_midis_pathlib / 'truth.mid')
        pianoroll.save_pianoroll(array=truth,
                                 path=self.save_midis_pathlib / 'truth.jpg',
                                 seed_length=self.nb_steps * self.step_length,
                                 instruments=self.instruments,
                                 mono=self.mono)
        # Missing instruments
        for inst in range(nb_instruments):
            midi.create.print_informations(nb_steps=self.nb_steps * self.step_length,
                                           matrix=filled_list[inst], notes_list=output_notes_inst[inst],
                                           verbose=verbose)
            # TODO: Accuracy
            midi.create.save_midi(output_notes_list=output_notes_inst[inst], instruments=self.instruments,
                                  path=self.save_midis_pathlib / f'missing_{inst}.mid')
            pianoroll.save_pianoroll(array=filled_list[inst],
                                     path=self.save_midis_pathlib / f'missing_{inst}.jpg',
                                     seed_length=self.nb_steps * self.step_length,
                                     instruments=self.instruments,
                                     mono=self.mono)

    def compare_generation(self, max_length=None, no_duration=False, verbose=1):
        """

        :return:
        """
        # -------------------- Find informations --------------------
        if self.data_transformed_pathlib is None:
            raise Exception('Some data need to be loaded before comparing the generation')
        nb_steps = int(self.model_id.split(',')[2])
        if self.my_sequence is None:
            self.my_sequence = Sequences.AllInstSequence(
                path=str(self.data_transformed_pathlib),
                nb_steps=nb_steps,
                batch_size=1,
                work_on=self.work_on)
        else:
            self.my_sequence.change_batch_size(1)
        self.my_sequence.set_noise(0)
        max_length = len(self.my_sequence) if max_length is None else min(max_length, len(self.my_sequence))

        # -------------------- Construct seeds --------------------
        generated = np.array(self.my_sequence[0][0])  # (nb_instrument, 1, nb_steps, step_size, input_size, 2) (1=batch)
        generated_helped = np.copy(generated)  # Each step will take the truth as an input
        generated_truth = np.copy(generated)  # The truth

        # -------------------- Generation --------------------
        bar = progressbar.ProgressBar(maxval=max_length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(max_length):
            ms_input, ms_output = self.my_sequence[l]
            sample = np.concatenate((generated[:, :, l:], np.array(ms_input)),
                                    axis=1)  # (nb_instruments, 2, nb_steps, step_size, input_size, 2)

            # Generation
            preds = self.keras_nn.generate(input=list(sample))

            # Reshape
            preds = np.asarray(preds).astype('float64')  # (nb_instruments, batch=2, nb_steps=1, length, 88, 2)
            preds_truth = np.array(ms_output)  # (nb_instruments, 1, 1, step_size, input_size, 2)
            # if only one instrument
            if len(preds.shape) == 5:  # Only one instrument : output of nn not a list
                preds = np.expand_dims(preds, axis=0)
            if len(preds_truth.shape) == 5:  # Only one instrument : output of nn not a list
                preds_truth = np.expand_dims(preds_truth)
            preds = midi.create.normalize_activation(preds, mono=self.mono)  # Normalize the activation part
            preds_helped = preds[:, [1]]  # (nb_instruments, 1, 1, length, 88, 2)
            preds = preds[:, [0]]

            # Concatenation
            generated = np.concatenate((generated, preds), axis=2)  # (nb_instruments, 1, nb_steps, length, 88, 2)
            generated_helped = np.concatenate((generated_helped, preds_helped),
                                              axis=2)  # (nb_instruments, 1, nb_steps, length, 88, 2)
            generated_truth = np.concatenate((generated_truth, preds_truth), axis=2)
            bar.update(l + 1)
        bar.finish()

        # -------------------- Compute notes list --------------------
        # Generated
        generated_midi_final = np.reshape(generated, (
            generated.shape[0], generated.shape[2] * generated.shape[3], generated.shape[4],
            generated.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final = np.transpose(generated_midi_final,
                                            (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        output_notes_list = midi.create.matrix_to_midi(generated_midi_final, instruments=self.instruments,
                                                       notes_range=self.notes_range, no_duration=no_duration,
                                                       mono=self.mono)
        # Helped
        generated_midi_final_helped = np.reshape(generated_helped, (
            generated_helped.shape[0], generated_helped.shape[2] * generated_helped.shape[3],
            generated_helped.shape[4], generated_helped.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final_helped = np.transpose(generated_midi_final_helped,
                                                   (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        output_notes_list_helped = midi.create.matrix_to_midi(generated_midi_final_helped, instruments=self.instruments,
                                                              notes_range=self.notes_range, no_duration=no_duration,
                                                              mono=self.mono)
        # Truth
        generated_midi_final_truth = np.reshape(generated_truth, (
            generated_truth.shape[0], generated_truth.shape[2] * generated_truth.shape[3], generated_truth.shape[4],
            generated_truth.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final_truth = np.transpose(generated_midi_final_truth,
                                                  (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        output_notes_list_truth = midi.create.matrix_to_midi(generated_midi_final_truth,
                                                             instruments=self.instruments,
                                                             notes_range=self.notes_range, no_duration=no_duration,
                                                             mono=self.mono)

        # ---------- find the name for the midi_file ----------
        self.get_new_save_midis_path()
        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)

        # -------------------- Save Results --------------------
        if self.work_on == 'note':
            step_length = 1
        elif self.work_on == 'beat':
            step_length = g.step_per_beat
        elif self.work_on == 'measure':
            step_length = 4 * g.step_per_beat
        # -- Generated --
        path_to_save = str(self.save_midis_pathlib / 'generated.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated.jpg')
        midi.create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final,
                                       notes_list=output_notes_list, verbose=verbose)
        # Print the accuracy
        if self.mono:
            argmax = np.argmax(generated_midi_final, axis=1)
            argmax_truth = np.argmax(generated_midi_final_truth, axis=1)
            accuracies = [(np.count_nonzero(
                argmax[i, nb_steps * step_length:] == argmax_truth[i,
                                                      nb_steps * step_length:]) / argmax[i,
                                                                                  nb_steps * step_length:].size)
                          for i in range(len(self.instruments))]
        else:
            accuracies = [((np.count_nonzero(
                generated_midi_final[i, :, nb_steps * step_length:, 0] == generated_midi_final_truth[i, :,
                                                                          nb_steps * step_length:,
                                                                          0])) / (generated_midi_final[i, :,
                                                                                  nb_steps * step_length:, 0].size)) for
                          i
                          in
                          range(len(self.instruments))]
        accuracy = sum(accuracies) / len(accuracies)
        print('Accuracy of the generation :', colored(accuracies, 'magenta'), ', overall :',
              colored(accuracy, 'magenta'))

        # Saving the Midi file
        midi.create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments,
                                 mono=self.mono)

        # -- Helped --
        path_to_save = str(self.save_midis_pathlib / 'generated_helped.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated_helped.jpg')
        midi.create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final_helped,
                                       notes_list=output_notes_list_helped, verbose=verbose)
        # Print the accuracy
        if self.mono:
            argmax_helped = np.argmax(generated_midi_final_helped, axis=1)
            argmax_truth = np.argmax(generated_midi_final_truth, axis=1)
            accuracies_helped = [(np.count_nonzero(
                argmax_helped[i, nb_steps * step_length:] == argmax_truth[i,
                                                             nb_steps * step_length:]) / argmax_helped[i,
                                                                                         nb_steps * step_length:].size)
                                 for i in range(len(self.instruments))]
        else:
            accuracies_helped = [(np.count_nonzero(
                generated_midi_final_helped[i, :, nb_steps * step_length:, 0] == generated_midi_final_truth[i, :,
                                                                                 nb_steps * step_length:,
                                                                                 0]) / generated_midi_final_helped[i, :,
                                                                                       nb_steps * step_length:, 0].size)
                                 for
                                 i in
                                 range(len(self.instruments))]
        accuracy_helped = sum(accuracies_helped) / len(accuracies_helped)

        print('Accuracy of the generation helped :', colored(accuracies_helped, 'magenta'), ', overall :',
              colored(accuracy_helped, 'magenta'))

        # Saving the Midi file
        midi.create.save_midi(output_notes_list=output_notes_list_helped, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final_helped,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments,
                                 mono=self.mono)

        # -- Truth --
        path_to_save = str(self.save_midis_pathlib / 'generated_truth.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated_truth.jpg')
        midi.create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final_truth,
                                       notes_list=output_notes_list_truth, verbose=verbose)

        # Saving the Midi file
        midi.create.save_midi(output_notes_list=output_notes_list_truth, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final_truth,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments,
                                 mono=self.mono)

        text = 'Generated :\n\tAccuracy : {0}, Accuracies : {1}\n'.format(accuracy, accuracies)
        text += 'Generated Helped :\n\tAccuracy : {0}, Accuracies : {1}'.format(accuracy_helped, accuracies_helped)
        with open(str(self.save_midis_pathlib / 'Results.txt'), 'w') as f:
            f.write(text)

        cprint('Done Generating', 'green')
