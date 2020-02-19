import random
from termcolor import colored, cprint
import numpy as np
import progressbar

from src.NN import Sequences
import src.Midi as midi
import src.text.summary as summary
from .MGInit import MGInit
from .MGComputeGeneration import MGComputeGeneration


class MGGenerate(MGComputeGeneration, MGInit):
    """

    """

    def generate_from_data(self, nb_seeds=10, length=None, new_save_path=None, save_images=False,
                           no_duration=False, verbose=1):
        """
        Generate Midi file from the seed and the trained model
        :param nb_seeds: number of seeds for the generation
        :param length: Length of th generation
        :param new_save_path:
        :param save_images: To save the pianoroll of the generation (.jpg images)
        :param no_duration: if True : all notes will be the shortest length possible
        :param verbose: Level of verbose
        :return:
        """
        # ---------- Verify the inputs ----------

        # ----- Create the seed -----
        if self.data_transformed_path is None:
            raise Exception('Some data need to be loaded before generating')
        self.sequence = Sequences.AllInstSequence(
            path=str(self.data_transformed_path),
            nb_steps=self.nb_steps,
            batch_size=1,
            work_on=self.work_on)
        nb_instruments = self.sequence.nb_instruments

        seeds_indexes = random.sample(range(len(self.sequence)), nb_seeds)

        # -- Length --
        length = length if length is not None else 10
        # --- Done Verifying the inputs ---
        mask = self.get_mask(nb_instruments)

        self.save_midis_path.mkdir(parents=True, exist_ok=True)
        cprint('Start generating from data ...', 'blue')
        for s in range(nb_seeds):
            cprint('Generation {0}/{1}'.format(s + 1, nb_seeds), 'blue')
            generated = np.array(
                self.sequence[seeds_indexes[s]][0])  # (nb_instruments, 1, nb_steps, step_size, inputs_size, 2)
            bar = progressbar.ProgressBar(maxval=length,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                                   progressbar.ETA()])
            bar.start()  # To see it working
            for l in range(length):
                samples = generated[:, :, l:]  # (nb_instruments, 1, nb_steps, length, 88, 2)   # 1 = batch
                # expanded_samples = np.expand_dims(samples, axis=0)
                preds = self.keras_nn.generate(
                    input=list(samples) + mask)  # (nb_instruments, batch=1 , nb_steps=1, length, 88, 2)
                preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, 1, step_size, input_size, 2)
                if len(preds.shape) == 4:  # Only one instrument : output of nn not a list
                    preds = preds[np.newaxis]
                next_array = midi.create.normalize_activation(preds)  # Normalize the activation part
                generated = np.concatenate((generated, next_array), axis=2)  # (nb_instruments, nb_steps, length, 88, 2)

                bar.update(l + 1)
            bar.finish()

            generated_midi_final = self.reshape_generated_array(generated)
            self.compute_generated_array(
                generated_array=generated_midi_final,
                folder_path=self.save_midis_path,
                name=f'generated_{s}',
                no_duration=no_duration,
                verbose=verbose,
                save_images=save_images
            )

        if self.batch is not None:
            self.sequence.batch_size = self.batch

        summary.summarize(
            # Function params
            path=self.save_midis_path,
            title=self.full_name,
            file_name='generate_summary.txt',
            # Summary params
            lenght=length,
            no_duration=no_duration,
            # Generic Summary
            **self.summary_dict
        )

        cprint('Done generating', 'green')

    def generate_fill(self, max_length=None, no_duration=False, verbose=1):
        """

        :param max_length:
        :param no_duration:
        :param verbose:
        :return:
        """
        # ----- Parameters -----
        max_length = 300 / self.step_length if max_length is None else max_length

        # ----- Variables -----
        if self.data_transformed_path is None:
            raise Exception('Some data need to be loaded before comparing the generation')
        sequence = Sequences.KerasSequence(
            path=self.data_transformed_path,
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
        cprint('Start generating (fill) ...', 'blue')
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
            preds = midi.create.normalize_activation(preds, mono=self.mono)
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
        truth = self.reshape_generated_array(truth)
        for inst in range(nb_instruments):
            filled_list[inst] = self.reshape_generated_array(filled_list[inst])
        self.save_midis_path.mkdir(parents=True, exist_ok=True)
        accuracies, accuracies_inst = self.compute_generated_array(
            generated_array=truth,
            folder_path=self.save_midis_path,
            name='generated_fill_truth',
            no_duration=no_duration,
            verbose=verbose,
            save_images=True
        )
        accuracies, accuracies_inst = [accuracies], [accuracies_inst]
        for inst in range(nb_instruments):
            acc, acc_inst = self.compute_generated_array(
                generated_array=filled_list[inst],
                folder_path=self.save_midis_path,
                name=f'generated_fill_{inst}',
                no_duration=no_duration,
                array_truth=truth,
                verbose=verbose,
                save_truth=False,
                save_images=True
            )
            accuracies.append(acc)
            accuracies_inst.append(acc_inst)

        # Save the image of all in a subplot to allow easier comparaisons
        self.save_generated_arrays_cross_images(
            generated_arrays=[truth] + filled_list,
            folder_path=self.save_midis_path,
            name=f'generated_fill_all',
            replicate=False,
            titles=['Truth'] + [f'Fill Inst {i}' for i in range(nb_instruments)],
            subtitles=[
                f'Acc: {accuracies_inst[i][int(max(0, i - 1))]}' for i in range(nb_instruments + 1)
            ]  # Truth is in it
        )

        # Save the summary of the generation
        summary.summarize(
            # Function parameters
            path=self.save_midis_path,
            title=self.full_name,
            file_name='generate_fill_summary.txt',
            # Summary parameters
            length=max_length,
            no_duration=no_duration,
            # Generic Summary
            **self.summary_dict
        )

        cprint('Done generating (fill)', 'green')

    def compare_generation(self, max_length=None, no_duration=False, verbose=1):
        """

        :return:
        """
        # -------------------- Find informations --------------------
        if self.data_transformed_path is None:
            raise Exception('Some data need to be loaded before comparing the generation')
        self.sequence = Sequences.AllInstSequence(
            path=str(self.data_transformed_path),
            nb_steps=self.nb_steps,
            batch_size=1,
            work_on=self.work_on,
            noise=0
        )
        max_length = len(self.sequence) if max_length is None else min(max_length, len(self.sequence))
        max_length = min(max_length, 10)

        # -------------------- Construct seeds --------------------
        generated = np.array(self.sequence[0][0])  # (nb_instrument, 1, nb_steps, step_size, input_size, 2) (1=batch)
        generated_helped = np.copy(generated)  # Each step will take the truth as an input
        generated_truth = np.copy(generated)  # The truth

        mask = self.get_mask(self.sequence.nb_instruments, batch_size=2)

        # -------------------- Generation --------------------
        cprint('Start comparing generation ...', 'blue')
        bar = progressbar.ProgressBar(maxval=max_length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(max_length):
            ms_input, ms_output = self.sequence[l]
            sample = np.concatenate((generated[:, :, l:], np.array(ms_input)),
                                    axis=1)  # (nb_instruments, 2, nb_steps, step_size, input_size, 2)

            # Generation
            preds = self.keras_nn.generate(input=list(sample) + mask)

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
        generated_midi_final = self.reshape_generated_array(generated)
        # Helped
        generated_midi_final_helped = self.reshape_generated_array(generated_helped)
        # Truth
        generated_midi_final_truth = self.reshape_generated_array(generated_truth)

        # ---------- find the name for the midi_file ----------
        self.save_midis_path.mkdir(parents=True, exist_ok=True)

        accuracies, accuracies_inst = [], []
        # Generated
        acc, acc_inst = self.compute_generated_array(
            generated_array=generated_midi_final,
            folder_path=self.save_midis_path,
            name='compare_generation_alone',
            no_duration=no_duration,
            array_truth=generated_midi_final_truth,
            verbose=verbose,
            save_truth=False,
            save_images=True
        )
        accuracies.append(acc)
        accuracies_inst.append(acc_inst)
        # Helped
        acc, acc_inst = self.compute_generated_array(
            generated_array=generated_midi_final_helped,
            folder_path=self.save_midis_path,
            name='compare_generation_helped',
            no_duration=no_duration,
            array_truth=generated_midi_final_truth,
            verbose=verbose,
            save_truth=False,
            save_images=True
        )
        accuracies.append(acc)
        accuracies_inst.append(acc_inst)
        # Truth
        self.compute_generated_array(
            generated_array=generated_midi_final_truth,
            folder_path=self.save_midis_path,
            name='compare_generation_truth',
            no_duration=no_duration,
            array_truth=None,
            verbose=verbose,
            save_truth=False,
            save_images=True
        )
        accuracies.append(acc)
        accuracies_inst.append(acc_inst)

        # Save the image of all in a subplot to allow easier comparaisons
        self.save_generated_arrays_cross_images(
            generated_arrays=[generated_midi_final_truth, generated_midi_final_helped, generated_midi_final],
            folder_path=self.save_midis_path,
            name=f'compare_generation_all',
            replicate=False,
            titles=['Truth', 'Helped', 'Alone'],
            subtitles=[
                'Acc : 1',
                f'Acc: {accuracies[1]:.3}, Acc_inst: [{", ".join([f"{a:.3}" for a in accuracies_inst[1]])}]',
                f'Acc: {accuracies[0]:.3}, Acc_inst: [{", ".join([f"{a:.3}" for a in accuracies_inst[0]])}]'
            ]  # Truth is in it
        )

        # ----- Summarize the generation -----

        # Creation of the summary .txt file
        summary.summarize(
            # Function parameters
            path=self.save_midis_path,
            title=self.full_name,
            file_name='compare_generation_summary.txt',
            # Summary paramters,
            length=max_length,
            no_duration=no_duration,
            generated_accuracy=accuracies[0],
            generated_accuracies=accuracies_inst[0],
            helped_accuracy=accuracies[1],
            helped_accuracies=accuracies_inst[1],
            # Generic Summary
            **self.summary_dict
        )

        cprint('Done comparing generation', 'green')

    def redo_song_generate(self, song_number=None, instrument_order=None, no_duration=False, save_images=True, noise=0):
        """

        :param instrument_order: The order of the instruments to remplace
        :param song_number: The number of the song in the dataset
        :param no_duration:
        :param save_images:
        :param noise:
        :return:
        """
        self.sequence = Sequences.KerasSequence(
            path=self.data_transformed_path,
            nb_steps=self.nb_steps,
            batch_size=1,
            work_on=self.work_on,
            noise=noise,
            replicate=False,
            predict_offset=self.predict_offset
        )
        song_number = np.random.randint(self.sequence.nb_songs) if song_number is None else song_number
        instrument_order = np.random.permutation(self.nb_instruments) if instrument_order is None else instrument_order
        all_arrays = []
        # Construct the truth array
        x, y = self.sequence.get_all_song(song_number=song_number, in_batch_format=False)
        # x: (nb_instruments, batch=1, nb_steps, step_size, input_size, channels)]
        # y: (nb_instruments, batch=1, nb_steps, step_size, input_size, channels)]
        # x and y are the same except that in x, there is some noise
        truth = x
        # truth: (nb_instruments, batch=1, len_song, step_size, input_size, channels)
        length = self.sequence.get_song_len(song_number)
        all_arrays.append(truth)
        cprint('Start redoing song (generate) ...', 'blue')
        bar = progressbar.ProgressBar(maxval=length * self.nb_instruments,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for instrument in range(len(instrument_order)):
            # We replace the instruments one by one
            instrument_to_remove = instrument_order[instrument]
            generated = np.copy(all_arrays[-1])
            # generated : (nb_instruments, batch=1, nb_steps, step_size, input_size, channels)
            generated[instrument_to_remove] = 0
            for step in range(length):
                inputs = np.take(generated, axis=2, indices=range(step, step + self.nb_steps))
                # inputs = (nb_instruments, batch=1, nb_steps, step_size, input_size, channels)]
                mask = self.get_mask()      # (batch=1, nb_instruments, nb_steps)
                # Remove the instrument from the input
                mask[0][:, instrument_to_remove] = 0
                inputs[instrument_to_remove] = 0
                preds = np.asarray(self.keras_nn.generate(input=list(inputs) + mask)).astype('float64')
                # preds: (nb_instruments, batch=1, nb_steps=1, step_size, input_size, channels)
                preds = midi.create.normalize_activation(preds, mono=self.mono)
                generated[instrument_to_remove, :, step:step + self.nb_steps] = preds[instrument_to_remove]

                bar.update(instrument * length + step)

            all_arrays.append(generated)
            # all_arrays: List(nb_instruments + 1)[(nb_instruments, batch=1, nb_steps, step_size, input_size, channels)]
        bar.finish()

        self.save_midis_path.mkdir(exist_ok=True, parents=True)
        generated_midi = [self.reshape_generated_array(arr) for arr in all_arrays]
        # Save the truth
        accuracies, accuracies_inst = self.compute_generated_array(
            generated_array=generated_midi[0],
            folder_path=self.save_midis_path,
            name='redo_song_generate_truth',
            no_duration=no_duration,
            save_images=save_images,
        )
        accuracies, accuracies_inst = [accuracies], [accuracies_inst]
        for inst in range(self.nb_instruments):
            acc, acc_inst = self.compute_generated_array(
                generated_array=generated_midi[inst + 1],
                folder_path=self.save_midis_path,
                name=f'redo_song_generate_{inst}_(inst_{instrument_order[inst]})',
                no_duration=no_duration,
                array_truth=generated_midi[0],
                save_images=save_images,
                save_truth=False,
            )
            accuracies.append(acc)
            accuracies_inst.append(acc_inst)

        if self.batch is not None:
            self.sequence.batch_size = self.batch

        self.save_generated_arrays_cross_images(
            generated_arrays=generated_midi,
            folder_path=self.save_midis_path,
            name='redo_song_all',
            titles=['Truth'] + [f'Iteration {i}: change inst {instrument_order[i]}' for i in
                                range(self.nb_instruments)],
            subtitles=[f'Acc: {accuracies[i]}, Acc inst: {accuracies_inst[i]}' for i in range(self.nb_instruments + 1)]
        )

        summary.summarize(
            # Function params
            path=self.save_midis_path,
            title=self.full_name,
            file_name='redo_song_replicate_summary.txt',
            # Summary params
            song_number=song_number,
            instrument_order=instrument_order,
            no_duration=no_duration,
            noise=noise,
            # Generic summary
            **self.summary_dict
        )

        cprint('Done redo song generate', 'green')
