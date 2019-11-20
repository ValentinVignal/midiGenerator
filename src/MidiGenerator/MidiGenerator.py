import os
from pathlib import Path

import src.global_variables as g
from .MGPredict import MGPredict
from .MGData import MGData
from .MGTrain import MGTrain
from .MGModel import MGModel
from .MGLogistic import MGLogistic


class MidiGenerator(MGPredict, MGData, MGTrain, MGModel, MGLogistic):
    """

    """

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
        self.save_midis_pathlib = None  # Where to save the generated midi files

        if data is not None:
            self.load_data(data)

    # --------------------------------------------------
    #               Class Methods
    # --------------------------------------------------

    @classmethod
    def from_model(cls, id, name='name', data=None):
        myModel = cls(name=name, data=data)
        myModel.load_model(id=id)
        return myModel

    @classmethod
    def with_new_model(cls, model_infos, name='name', work_on=g.work_on, data=None):
        myModel = cls(name=name, data=data)

        def get_value(key):
            """

            :param key: key in the dictionary "model_infos"
            :return: the value in model_infos or None if it doesn't exist
            """
            value = None if key not in model_infos else model_infos[key]
            return value

        myModel.input_param = model_infos['input_param']
        myModel.model_id = model_infos['model_id']
        myModel.new_nn_model(
            model_id=model_infos['model_id'],
            work_on=work_on,
            opt_param=get_value('opt_param'),
        )
        return myModel

    @classmethod
    def with_model(cls, id, with_weights=True):
        my_model = cls()
        my_model.recreate_model(id=id, with_weigths=with_weights)
        return my_model

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
        return g.work_on2nb(self.work_on)

