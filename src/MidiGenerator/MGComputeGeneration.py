from termcolor import colored, cprint
import numpy as np
from pathlib import Path

import src.Midi as midi
from src.NN import Models
from src.Images import pianoroll
from .MGInit import MGInit


class MGComputeGeneration(MGInit):
    """

    """

    @staticmethod
    def reshape_generated_array(array):
        """

        :param array: (nb_instruments, batch=1, nb_steps, step_length, size, channels)
        :return: array: (nb_instruments, size, length, channels)
        """
        array = np.reshape(
            array,
            (array.shape[0], array.shape[2] * array.shape[3], *array.shape[4:])
        )  # (nb_instruments, length, size, channels)
        array = np.transpose(
            array,
            (0, 2, 1, 3)
        )  # (nb_instruments, size, length, channels)
        return array

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

    def compute_generated_array(self, generated_array, folder_path, name, no_duration=False, array_truth=None,
                                verbose=1,
                                save_truth=False, save_images=True, replicate=False):
        """

        :param replicate:
        :param save_images:
        :param save_truth:
        :param verbose:
        :param array_truth:
        :param generated_array:
        :param folder_path
        :param name
        :param no_duration:
        """
        folder_path = Path(folder_path)

        # if files exist -> find new name
        if (folder_path / name).with_suffix('.mid').exists() or \
                (folder_path / (name + '_(PIL)')).with_suffix('.jpg').exists() or \
                (folder_path / (name + '_(PLT)')).with_suffix('.jpg').exists():
            i = 0
            while (folder_path / (name + f'_({i}).mid')).exists() or \
                    (folder_path / (name + f'_({i})_(PIL).jpg')).exists() or \
                    (folder_path / (name + f'_({i})_(PLT).jpg')).exists():
                i += 1
            name += f'_({i})'

        midi_path = (folder_path / name).with_suffix('.mid')

        output_notes = midi.create.matrix_to_midi(generated_array,
                                                  instruments=self.instruments,
                                                  notes_range=self.notes_range, no_duration=no_duration,
                                                  mono=self.mono)
        midi.create.print_informations(nb_steps=self.nb_steps * self.step_length,
                                       matrix=generated_array, notes_list=output_notes, verbose=verbose)
        midi.create.save_midi(output_notes_list=output_notes, instruments=self.instruments,
                              path=midi_path)
        if save_images:
            pianoroll.save_array_as_pianoroll(array=generated_array,
                                              folder_path=folder_path,
                                              name=name,
                                              seed_length=self.nb_steps * self.step_length,
                                              mono=self.mono,
                                              replicate=replicate)
        if array_truth is not None:
            accuracy, accuracies_inst = self.accuracy_generation(generated_array, array_truth, mono=self.mono)
            print(f'Accuracy of the generation {name} :', colored(accuracies_inst, 'magenta'), ', overall :',
                  colored(accuracy, 'magenta'))
            if save_truth:
                output_notes_truth = midi.create.matrix_to_midi(array_truth,
                                                                instruments=self.instruments,
                                                                notes_range=self.notes_range, no_duration=no_duration,
                                                                mono=self.mono)
                midi.create.save_midi(output_notes_list=output_notes_truth, instruments=self.instruments,
                                      path=folder_path / f'{name}_truth.mid')
                if save_images:
                    pianoroll.save_array_as_pianoroll(array=array_truth,
                                                      folder_path=folder_path,
                                                      name=f'{name}_truth',
                                                      seed_length=self.nb_steps * self.step_length,
                                                      mono=self.mono,
                                                      replicate=replicate)

    def get_mask(self, nb_instruments, batch_size=1):
        if Models.needs_mask[self.model_name]:
            mask = [np.ones((batch_size, nb_instruments, self.nb_steps))]
        else:
            mask = []
        return mask
