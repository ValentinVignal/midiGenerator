import os
from pathlib import Path
from src import GlobalVariables as g


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
        self.predict_offset = None
        self.get_new_full_name()

        self.saved_model_path = Path(
            os.path.join('saved_models', self.full_name))  # Where to saved the trained model

        # ----- Data -----
        self.data_transformed_path = None

        self.instruments = None  # List of instruments used
        self.notes_range = None

        # ----- MySequence -----
        self.sequence = None  # Instance of MySequence Generator
        self.batch = None  # Size if the batch
        self.mono = None  # If this is not polyphonic instrument and no rest

        # ----- Neural Network -----
        self.input_param = None  # The parameters for the neural network
        self.keras_nn = None  # Our neural network
        self.train_history = None

        # ------ save_midi_path -----
        self.save_midis_path = None  # Where to save the generated Midi files

        if data is not None:
            self.load_data(data)

        # --------------------------------------------------
        #                   Properties
        # --------------------------------------------------

    @property
    def nb_steps(self):
        """

        :return:
        """
        return int(self.model_id.split(',')[2])

    @property
    def step_length(self):
        return g.mg.work_on2nb(self.work_on)

    @property
    def model_name(self):
        return self.model_id.split(',')[0]

    def __del__(self):
        del self.keras_nn
        del self.sequence
        del self.train_history

    @property
    def nb_instruments(self):
        return len(self.instruments)

    @property
    def summary_dict(self):
        """

        :return: A dict with all the interesting attribut of the instance
        """
        return dict(
            epochs=self.total_epochs,
            name=self.name,
            model_id=self.model_id,
            full_name=self.full_name,
            work_on=self.work_on,
            predict_offset=self.predict_offset,
            nb_instruments=self.nb_instruments,
            instrumetns=self.instruments,
            notes_range=self.notes_range,
            mono=self.mono
        )

