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

    def replicate_fom_data(self, length=None, new_save_path=None, save_images=False,
                           no_duration=False, verbose=1, noise=0):
        """
        Generate Midi file from the seed and the trained model
        :param length: Length of th generation
        :param new_save_path:
        :param save_images: To save the pianoroll of the generation (.jpg images)
        :param no_duration: if True : all notes will be the shortest length possible
        :param verbose: Level of verbose
        :param noise:
        :return:
        """
        # ---------- Verify the inputs ----------

        # ----- Create the seed -----
        if self.data_transformed_path is None:
            raise Exception('Some data need to be loaded before generating')
        self.sequence = Sequences.AllInstSequenceReplicate(
            path=str(self.data_transformed_path),
            nb_steps=self.nb_steps,
            batch_size=1,
            work_on=self.work_on,
            noise=noise
        )

        # -- Length --
        length = length if length is not None else min(20, len(self.sequence))
        # -- For save Midi path --
        if type(new_save_path) is str or (
                type(new_save_path) is bool and new_save_path) or (
                new_save_path is None and self.save_midis_path is None):
            self.get_new_save_midis_path(path=new_save_path)
        # --- Done Verifying the inputs ---

        self.save_midis_path.mkdir(parents=True, exist_ok=True)

        cprint('Start generating ...', 'blue')

        shape_with_no_step = list(np.array(self.sequence[0][0]).shape)
        shape_with_no_step[2] = 0
        shape_with_no_step = tuple(shape_with_no_step)
        generated = np.zeros(
            shape=shape_with_no_step)  # (nb_instruments, batch=1, nb_steps=0, step_size, inputs_size, 2)
        truth = np.zeros(
            shape=shape_with_no_step)  # (nb_instruments, batch=1, nb_steps=0, step_size, inputs_size, 2)
        bar = progressbar.ProgressBar(maxval=length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(0, length, self.nb_steps):
            x, y = self.sequence[l]

            preds = self.keras_nn.generate(
                input=x)  # (nb_instruments, batch=1 , nb_steps=1, length, 88, 2)
            preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, 1, step_size, input_size, 2)
            if len(preds.shape) == 4:  # Only one instrument : output of nn not a list
                preds = preds[np.newaxis]
            next_array = midi.create.normalize_activation(preds, mono=self.mono)  # Normalize the activation part
            generated = np.concatenate((generated, next_array), axis=2)  # (nb_instruments, 1, nb_steps, length, 88, 2)
            truth = np.concatenate((truth, np.asarray(y)),
                                   axis=2)  # (nb_instruments, 1, nb_steps, step_length, size, channels)

            bar.update(l + 1)
        bar.finish()

        """
        generated_midi_final = np.reshape(generated, (
            generated.shape[0], generated.shape[2] * generated.shape[3], generated.shape[4],
            generated.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final = np.transpose(generated_midi_final,
                                            (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        """
        generated_midi_final = self.reshape_generated_array(generated)
        truth_final = self.reshape_generated_array(truth)
        self.compute_generated_array(
            generated_array=generated_midi_final,
            file_name=self.save_midis_path / f'out',
            no_duration=no_duration,
            array_truth=truth_final,
            verbose=verbose,
            save_images=save_images,
            save_truth=True
        )

        if self.batch is not None:
            self.sequence.change_batch_size(self.batch)

        summary.summarize_generation(str(self.save_midis_path), **{
            'full_name': self.full_name,
            'epochs': self.total_epochs,
            'input_param': self.input_param,
            'instruments': self.instruments,
            'notes_range': self.notes_range
        })

        cprint('Done Generating', 'green')

    def compare_generation(self, max_length=None, no_duration=False, verbose=1):
        """

        :return:
        """
        # -------------------- Find informations --------------------
        if self.data_transformed_path is None:
            raise Exception('Some data need to be loaded before comparing the generation')
        nb_steps = int(self.model_id.split(',')[2])
        if self.sequence is None:
            self.sequence = Sequences.AllInstSequence(
                path=str(self.data_transformed_path),
                nb_steps=nb_steps,
                batch_size=1,
                work_on=self.work_on)
        else:
            self.sequence.change_batch_size(1)
        self.sequence.set_noise(0)
        max_length = len(self.sequence) if max_length is None else min(max_length, len(self.sequence))

        # -------------------- Construct seeds --------------------
        generated = np.array(self.sequence[0][0])  # (nb_instrument, 1, nb_steps, step_size, input_size, 2) (1=batch)
        generated_helped = np.copy(generated)  # Each step will take the truth as an input
        generated_truth = np.copy(generated)  # The truth

        # -------------------- Generation --------------------
        bar = progressbar.ProgressBar(maxval=max_length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(max_length):
            ms_input, ms_output = self.sequence[l]
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
        # Helped
        generated_midi_final_helped = np.reshape(generated_helped, (
            generated_helped.shape[0], generated_helped.shape[2] * generated_helped.shape[3],
            generated_helped.shape[4], generated_helped.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final_helped = np.transpose(generated_midi_final_helped,
                                                   (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        # Truth
        generated_midi_final_truth = np.reshape(generated_truth, (
            generated_truth.shape[0], generated_truth.shape[2] * generated_truth.shape[3], generated_truth.shape[4],
            generated_truth.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final_truth = np.transpose(generated_midi_final_truth,
                                                  (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)

        # ---------- find the name for the midi_file ----------
        self.get_new_save_midis_path()
        self.save_midis_path.mkdir(parents=True, exist_ok=True)

        # Generated
        self.compute_generated_array(
            generated_array=generated_midi_final,
            file_name=self.save_midis_path / 'generated',
            no_duration=no_duration,
            array_truth=generated_midi_final_truth,
            verbose=verbose,
            save_truth=False,
            save_images=True
        )
        # Helped
        self.compute_generated_array(
            generated_array=generated_midi_final_helped,
            file_name=self.save_midis_path / 'helped',
            no_duration=no_duration,
            array_truth=generated_midi_final_truth,
            verbose=verbose,
            save_truth=False,
            save_images=True
        )
        # Truth
        self.compute_generated_array(
            generated_array=generated_midi_final_truth,
            file_name=self.save_midis_path / 'truth',
            no_duration=no_duration,
            array_truth=None,
            verbose=verbose,
            save_truth=False,
            save_images=True
        )
        cprint('Done Generating', 'green')
