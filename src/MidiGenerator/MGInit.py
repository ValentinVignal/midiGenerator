import os
from pathlib import Path


class MGInit:
    def __init__(self, name='name', data=None):
        """

        :param name: The name of the model
        :param work_on: either 'beat' or 'measure'
        :param data: if not None, load the data
        """
        # ----- General -----
        self.total_epochs = 0
        self.name = name
        self.model_id = ''  # Id of the model used
        self.full_name = ''  # Id of this MyModel instance
        self.work_on = None
        self.get_new_full_name()

        self.saved_model_pathlib = Path(
            os.path.join('saved_models', self.full_name))  # Where to saved the trained model

        # ----- Data -----
        self.data_transformed_pathlib = None

        self.instruments = None  # List of instruments used
        self.notes_range = None

        # ----- MySequence -----
        self.my_sequence = None  # Instance of MySequence Generator
        self.batch = None  # Size if the batch
        self.mono = None  # If this is not polyphonic instrument and no rest

        # ----- Neural Network -----
        self.input_param = None  # The parameters for the neural network
        self.keras_nn = None  # Our neural network
        self.train_history = None

        # ------ save_midi_path -----
        self.save_midis_pathlib = None  # Where to save the generated Midi files

        if data is not None:
            self.load_data(data)
