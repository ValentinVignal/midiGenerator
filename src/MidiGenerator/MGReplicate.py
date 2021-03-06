from termcolor import colored, cprint
import numpy as np
import progressbar

from src.NN import Sequences
import src.Midi as midi
import src.text.summary as summary
from .MGInit import MGInit
from .MGComputeGeneration import MGComputeGeneration


class MGReplicate(MGComputeGeneration, MGInit):
    """

    """

    def replicate_from_data(self, length=None, save_images=False, no_duration=False, verbose=1, noise=0):
        """
        Generate Midi file from the seed and the trained model
        :param length: Length of th generation
        :param save_images: To save the pianoroll of the generation (.jpg images)
        :param no_duration: if True : all notes will be the shortest length possible
        :param verbose: Level of verbose
        :param noise:
        :return:
        """
        # ---------- Verify the inputs ----------

        # ----- Create the seed -----
        if self.data_transformed_path is None:
            raise Exception('Some data need to be loaded before replicating')
        self.sequence = Sequences.AllInstSequence.replicate(
            path=str(self.data_transformed_path),
            nb_steps=self.nb_steps,
            batch_size=1,
            work_on=self.work_on,
            noise=noise
        )
        nb_instruments = self.sequence.nb_instruments

        # -- Length --
        length = length if length is not None else min(20, len(self.sequence))
        # --- Done Verifying the inputs ---

        self.save_midis_path.mkdir(parents=True, exist_ok=True)

        shape_with_no_step = list(np.array(self.sequence[0][0]).shape)
        shape_with_no_step[2] = 0
        shape_with_no_step = tuple(shape_with_no_step)
        generated = np.zeros(
            shape=shape_with_no_step)  # (nb_instruments, batch=1, nb_steps=0, step_size, inputs_size, 2)
        truth = np.zeros(
            shape=shape_with_no_step)  # (nb_instruments, batch=1, nb_steps=0, step_size, inputs_size, 2)
        mask = self.get_mask(nb_instruments)

        cprint('Start replicating ...', 'blue')
        bar = progressbar.ProgressBar(maxval=length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(0, length, self.nb_steps):
            x, y = self.sequence[l]
            y = np.asarray(y)

            preds = self.keras_nn.generate(
                input=x + mask)  # (nb_instruments, batch=1 , nb_steps=1, length, 88, 2)
            preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, 1, step_size, input_size, 2)
            if len(preds.shape) == 5:  # Only one instrument : output of nn not a list
                preds = preds[np.newaxis]
            if len(y.shape) == 5:
                y = np.expand_dims(y, axis=0)
            next_array = midi.create.normalize_activation(
                preds,
                mono=self.mono,
                use_binary=self.use_binary)  # Normalize the activation part
            generated = np.concatenate((generated, next_array), axis=2)  # (nb_instruments, 1, nb_steps, length, 88, 2)
            truth = np.concatenate((truth, y),
                                   axis=2)  # (nb_instruments, 1, nb_steps, step_length, size, channels)

            bar.update(l + 1)
        bar.finish()

        generated_midi_final = self.reshape_generated_array(generated)
        truth_final = self.reshape_generated_array(truth)
        self.compute_generated_array(
            generated_array=generated_midi_final,
            folder_path=self.save_midis_path,
            name=f'replicated',
            no_duration=no_duration,
            array_truth=truth_final,
            verbose=verbose,
            save_images=save_images,
            save_truth=True,
            replicate=True
        )

        if self.batch is not None:
            self.sequence.batch_size = self.batch

        summary.summarize(
            # Function params
            path=self.save_midis_path,
            title=self.full_name,
            file_name='replicate_summary.txt',
            # Summary params
            length=length,
            no_duration=no_duration,
            noise=noise,
            # Generic Summary
            **self.summary_dict
        )

        cprint('Done replicating', 'green')

    def replicate_fill(self, max_length=None, no_duration=False, verbose=1, save_images=True, noise=0):
        """

        :param max_length:
        :param no_duration:
        :param verbose:
        :param save_images:
        :param noise:
        :return:
        """
        max_length = max_length if max_length is not None else 300 / self.step_length
        if self.data_transformed_path is None:
            raise Exception('Some data need to be loaded before replicating')
        self.sequence = Sequences.KerasSequence(
            path=str(self.data_transformed_path),
            nb_steps=self.nb_steps,
            batch_size=1,
            work_on=self.work_on,
            noise=noise,
            replicate=True
        )
        max_length = int(min(max_length, len(self.sequence)))
        nb_instruments = self.sequence.nb_instruments

        self.save_midis_path.mkdir(parents=True, exist_ok=True)

        shape_with_no_step = list(np.array(self.sequence[0][0]).shape)
        shape_with_no_step[2] = 0
        shape_with_no_step = tuple(shape_with_no_step)
        generated_list = [
            np.zeros(
                shape=shape_with_no_step
            )  # (nb_instruments, batch=1, nb_steps=0, step_size, inputs_size, 2)
            for inst in range(nb_instruments)
        ]
        truth = np.zeros(
            shape=shape_with_no_step)  # (nb_instruments, batch=1, nb_steps=0, step_size, inputs_size, 2)
        mask = np.ones((nb_instruments, nb_instruments, self.nb_steps))
        for inst in range(nb_instruments):
            mask[inst, inst] = 0

        cprint('Start replicating (fill) ...', 'blue')
        bar = progressbar.ProgressBar(maxval=max_length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(0, max_length, self.nb_steps):
            x, y = self.sequence[l]
            x_missing_inst_list = []
            for inst in range(nb_instruments):
                x_missing_inst = np.copy(x)
                x_missing_inst[inst] = 0  # (nb_instruments, batch, nb_steps, step_length, size, channels)
                x_missing_inst_list.append(x_missing_inst)
            nn_input = np.concatenate(
                tuple(x_missing_inst_list),
                axis=1
            )  # (nb_instruments, batch=nb_instruments, nb_steps, step_length, size, channels)
            preds = self.keras_nn.generate(
                input=list(nn_input) + [mask])  # (nb_instruments, batch=1 , nb_steps=1, length, 88, 2)
            preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, 1, step_size, input_size, 2)
            if len(preds.shape) == 5:  # Only one instrument : output of nn not a list
                preds = preds[np.newaxis]
            if len(y.shape) == 5:
                y = np.expand_dims(y, axis=0)
            preds = midi.create.normalize_activation(
                preds,
                mono=self.mono,
                use_binary=self.use_binary
            )  # Normalize the activation part
            for inst in range(nb_instruments):
                p = np.copy(y)  # (nb_instruments, batch=1, nb_steps, step_length, size, channels)
                p[inst] = np.take(preds, axis=1, indices=[inst])[inst]
                generated_list[inst] = np.concatenate(
                    (generated_list[inst], p),
                    axis=2
                )  # (nb_instruments, batch=1, nb_steps, step_length, size, channels)
            truth = np.concatenate((truth, np.asarray(y)),
                                   axis=2)  # (nb_instruments, 1, nb_steps, step_length, size, channels)
            bar.update(l + 1)
        bar.finish()

        generated_midi_final_list = [
            self.reshape_generated_array(generated_list[inst]) for inst in range(nb_instruments)
        ]
        truth_final = self.reshape_generated_array(truth)
        accuracies, accuracies_inst = self.compute_generated_array(
            generated_array=truth_final,
            folder_path=self.save_midis_path,
            name='replicated_fill_truth',
            no_duration=no_duration,
            verbose=verbose,
            save_images=save_images,
            replicate=True
        )
        accuracies, accuracies_inst = [accuracies], [accuracies_inst]
        for inst in range(nb_instruments):
            acc, acc_inst = self.compute_generated_array(
                generated_array=generated_midi_final_list[inst],
                folder_path=self.save_midis_path,
                name=f'replicated_fill_{inst}',
                no_duration=no_duration,
                array_truth=truth_final,
                verbose=verbose,
                save_images=save_images,
                save_truth=False,
                replicate=True
            )
            accuracies.append(acc)
            accuracies_inst.append(acc_inst)

        if self.batch is not None:
            self.sequence.batch_size = self.batch

        # Save the image of all in a subplot to allow easier comparaisons
        self.save_generated_arrays_cross_images(
            generated_arrays=[truth_final] + generated_midi_final_list,
            folder_path=self.save_midis_path,
            name=f'replicated_fill_all',
            replicate=True,
            titles=['Truth'] + [f'Fill Inst {i}' for i in range(nb_instruments)],
            subtitles=[
                f'Acc: {accuracies_inst[i][int(max(0, i - 1))]}' for i in range(nb_instruments + 1)
            ]  # Truth is in it
        )

        # Save the summary of the generation
        summary.summarize(
            # Function params
            path=self.save_midis_path,
            title=self.full_name,
            file_name='replicate_fill_summary.txt',
            # Summary params
            length=max_length,
            no_duration=no_duration,
            noise=noise,
            # Generic Summary
            **self.summary_dict
        )

        cprint('Done replicating (fill)', 'green')

    def redo_song_replicate(self, song_number=None, instrument_order=None, no_duration=False, save_images=True,
                            noise=0):
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
            replicate=True
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
        # truth: (nb_instruments, batch=1, len_song, step_size, input_size, channels
        # We then take a number of step which is a multiple of self.nb_steps
        indices = range(truth.shape[2] // self.nb_steps * self.nb_steps)
        truth = np.take(truth, axis=2, indices=indices)
        length = truth.shape[2]
        all_arrays.append(truth)
        cprint('Start redoing song (replicate) ...', 'blue')
        bar = progressbar.ProgressBar(maxval=length * self.nb_instruments,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for instrument in range(len(instrument_order)):
            # We replace the instruments one by one
            instrument_to_remove = instrument_order[instrument]
            generated = []
            for step in range(0, length, self.nb_steps):
                inputs = np.take(all_arrays[-1], axis=2, indices=range(step, step + self.nb_steps))
                # inputs = (nb_instruments, batch=1, nb_steps, step_size, input_size, channels)]
                mask = np.ones((1, self.nb_instruments, self.nb_steps))  # (batch=1, nb_instruments, nb_steps)
                # Remove the instrument from the input
                mask[:, instrument_to_remove] = 0
                inputs[instrument_to_remove] = 0
                preds = np.asarray(self.keras_nn.generate(input=list(inputs) + [mask])).astype('float64')
                # preds: (nb_instruments, batch=1, nb_steps, step_size, input_size, channels)
                preds = midi.create.normalize_activation(
                    preds,
                    mono=self.mono,
                    use_binary=self.use_binary
                )
                outputs = np.copy(inputs)
                outputs[instrument_to_remove] = preds[instrument_to_remove]
                generated.append(outputs)

                bar.update(instrument * length + step)

            all_arrays.append(np.concatenate(generated, axis=2))
            # all_arrays: List(nb_instruments + 1)[(nb_instruments, batch=1, nb_steps, step_size, input_size, channels)]
        bar.finish()

        self.save_midis_path.mkdir(exist_ok=True, parents=True)
        generated_midi = [self.reshape_generated_array(arr) for arr in all_arrays]
        # Save the truth
        accuracies, accuracies_inst = self.compute_generated_array(
            generated_array=generated_midi[0],
            folder_path=self.save_midis_path,
            name='redo_song_replicate_truth',
            no_duration=no_duration,
            save_images=save_images,
            replicate=True
        )
        accuracies, accuracies_inst = [accuracies], [accuracies_inst]
        for inst in range(self.nb_instruments):
            acc, acc_inst = self.compute_generated_array(
                generated_array=generated_midi[inst + 1],
                folder_path=self.save_midis_path,
                name=f'redo_song_replicate_{inst}_(inst_{instrument_order[inst]})',
                no_duration=no_duration,
                array_truth=generated_midi[0],
                save_images=save_images,
                save_truth=False,
                replicate=True
            )
            accuracies.append(acc)
            accuracies_inst.append(acc_inst)

        if self.batch is not None:
            self.sequence.batch_size = self.batch

        self.save_generated_arrays_cross_images(
            generated_arrays=generated_midi,
            folder_path=self.save_midis_path,
            name='redo_song_all',
            replicate=True,
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

        cprint('Done redo song replicate', 'green')
