from termcolor import colored, cprint
import numpy as np
from pathlib import Path

import src.Midi as midi
import src.image.pianoroll as pianoroll
from .MGInit import MGInit


class MGComputeGeneration(MGInit):
    """

    """
    @staticmethod
    def accuracy_generation(array, truth, mono=False):
        """

        :param array: (nb_instruments, size, nb_step * length,  channels)
        :param truth: (nb_instruments, size, nb_step * length,  channels)
        :param mono: boolean
        :return: accuracy of the generation
        """
        if mono:
            argmax = np.argmax(array, axis=1)  # (nb_instruments, length, channels)
            argmax_truth = np.argmax(truth, axis=1)  # (nb_instruments, length, channels)
            accuracies_inst = np.count_nonzero(argmax == argmax_truth, axis=(1, 2)) / argmax[0].size
        else:
            accuracies_inst = np.count_nonzero(array == truth, axis=(1, 2, 3)) / array[0].size
        accuracy = np.mean(accuracies_inst)
        return accuracy, list(accuracies_inst)

    def compute_generated_array(self, generated_array, file_name, no_duration=False, array_truth=None, verbose=1,
                                save_truth=False, save_images=True):
        """

        :param save_images:
        :param save_truth:
        :param verbose:
        :param array_truth:
        :param generated_array:
        :param file_name:
        :param no_duration:
        """
        file_name = Path(file_name)
        # if files exist -> find new name
        if file_name.with_suffix('.mid').exists() or file_name.with_suffix('.jpg').exists():
            i = 0
            while file_name.with_suffix(f'_({i}).mid').exists() or file_name.with_suffix(f'_({i}).jpg').exists():
                i += 1
            file_name = file_name.with_suffix(f'_({i})')
        name = file_name.name
        midi_path = file_name.with_suffix('.mid')
        img_path = file_name.with_suffix('.jpg')

        output_notes = midi.create.matrix_to_midi(generated_array,
                                                  instruments=self.instruments,
                                                  notes_range=self.notes_range, no_duration=no_duration,
                                                  mono=self.mono)
        midi.create.print_informations(nb_steps=self.nb_steps * self.step_length,
                                       matrix=generated_array, notes_list=output_notes, verbose=verbose)
        midi.create.save_midi(output_notes_list=output_notes, instruments=self.instruments,
                              path=midi_path)
        if save_images:
            pianoroll.save_pianoroll(array=generated_array,
                                     path=img_path,
                                     seed_length=self.nb_steps * self.step_length,
                                     instruments=self.instruments,
                                     mono=self.mono)
        if array_truth is not None:
            accuracy, accuracies_inst = self.accuracy_generation(generated_array, array_truth, mono=self.mono)
            print(f'Accuracy of the generation {name} :', colored(accuracies_inst, 'magenta'), ', overall :',
                  colored(accuracy, 'magenta'))
            if save_truth:
                output_notes_truth = midi.create.matrix_to_midi(generated_array,
                                                                instruments=self.instruments,
                                                                notes_range=self.notes_range, no_duration=no_duration,
                                                                mono=self.mono)
                midi.create.save_midi(output_notes_list=output_notes_truth, instruments=self.instruments,
                                      path=file_name.with_suffix('_truth.mid'))
                if save_images:
                    pianoroll.save_pianoroll(array=generated_array,
                                             path=file_name.with_suffix('_truth.jpg'),
                                             seed_length=self.nb_steps * self.step_length,
                                             instruments=self.instruments,
                                             mono=self.mono)
