from epicpath import EPath
import warnings

from src import GlobalVariables as g


class MGInit:
    def __init__(self, name='name', data=None):
        """

        :param name: The name of the model
        :param work_on: either 'beat' or 'measure'
        :param data: if not None, load the data
        """
        # ----- General -----
        self._total_epochs = 0
        self._name = name
        self._model_id = ''  # Id of the model used
        self._full_name_i = 0       # To complete the full name
        self.work_on = None
        self.predict_offset = None

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
        self._save_midis_path_i = None  # Where to save the generated Midi files

        self.get_new_i()

        if data is not None:
            self.load_data(data)

        # --------------------------------------------------
        #                   Properties
        # --------------------------------------------------

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        self.delete_tokens()
        self._model_id = model_id
        self.get_new_i()

    @property
    def total_epochs(self):
        return self._total_epochs

    @total_epochs.setter
    def total_epochs(self, total_epochs):
        self.delete_tokens()
        self._total_epochs = total_epochs
        self.get_new_i()

    @property
    def full_name_i(self):
        if self._full_name_i is None:
            self.get_new_i()
        return self._full_name_i

    @full_name_i.setter
    def full_name_i(self, i):
        self.delete_tokens()
        self._full_name_i = i
        self.create_tokens()

    @full_name_i.deleter
    def full_name_i(self):
        self.delete_token()
        del self._full_name_i

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self.delete_tokens()
        self._name = self._name if name is None else name
        self.get_new_i()

    @property
    def full_name(self):
        return f'{self.full_name_no_i}-({self.full_name_i})'

    @property
    def saved_model_path(self):
        return EPath('saved_models', self.full_name)

    @property
    def full_name_no_i(self):
        return f'{self.name}-m({self.model_id})-e({self.total_epochs})'

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

    @property
    def save_midis_path_i(self):
        if self._save_midis_path_i is None:
            self.get_new_save_midis_path_i()
        return self._save_midis_path_i

    @save_midis_path_i.setter
    def save_midis_path_i(self, save_midis_path_i):
        self.delete_token_midis_path()
        self._save_midis_path_i = save_midis_path_i
        self.create_token_midis_path()

    @save_midis_path_i.deleter
    def save_midis_path_i(self):
        self.delete_token_midis_path()
        del self._save_midis_path_i

    @property
    def save_midis_path(self):
        return EPath('generated_midis', f'{self.full_name}-generation({self.save_midis_path_i})')

    @property
    def input_size(self):
        return self.input_param['input_size']

    def __del__(self, *args, **kwargs):
        del self.keras_nn
        del self.sequence
        del self.train_history
        self.delete_tokens()

    # ----------------------------------------------------------------------------------------------------
    #                                           Functions
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def warning_init_function(function_name=None, subclass=None):
        message = 'Call of a function in MGInit '
        if function_name is not None:
            message += f'"{function_name}" '
        message += ', it does nothing, this function should be overwritten by a subclass'
        if subclass is not None:
            message += f'("{subclass}")'
        warnings.warn(message)

    # ---------------------------------------- MGComputeGeneration ----------------------------------------

    @staticmethod
    def reshape_generated_array(*args, **kwargs):
        MGInit.warning_init_function(MGInit.reshape_generated_array.__name__, 'MGComputeGeneration')

    @staticmethod
    def accuracy_generation(*args, **kwargs):
        MGInit.warning_init_function(MGInit.accuracy_generation.__name__, 'MGComputeGeneration')

    def accuracy_generation(self, *args, **kwargs):
        self.warning_init_function(self.accuracy_generation.__name__, 'MGComputeGeneration')

    def compute_generated_array(self, *args, **kwargs):
        self.warning_init_function(self.compute_generated_array.__name__, 'MGComputeGeneration')

    def save_generated_arrays_cross_images(self, *args, **kwargs):
        self.warning_init_function(self.save_generated_arrays_cross_images.__name__, 'MGComputeGeneration')

    def get_mask(self, *args, **kwargs):
        self.warning_init_function(self.get_mask.__name__, 'MGCompute_generation')

    # ---------------------------------------- MGData ----------------------------------------

    def load_data(self, *args, **kwargs):
        self.warning_init_function(self.load_data.__name__, 'MGData')

    def change_batch_size(self, *args, **kwargs):
        self.warning_init_function(self.change_batch_size.__name__, 'MGData')

    def get_sequence(self, *args, **kwargs):
        self.warning_init_function(self.get_sequence.__name__, 'MGData')

    # ---------------------------------------- MGGenerate ----------------------------------------

    def generate_from_data(self, *args, **kwargs):
        self.warning_init_function(self.generate_from_data.__name__, 'MGGenerate')

    def generate_fill(self, *args, **kwargs):
        self.warning_init_function(self.generate_fill.__name__, 'MGGenerate')

    def compare_generation(self, *args, **kwargs):
        self.warning_init_function(self.compare_generation.__name__, 'MGGenerate')

    def redo_song_generate(self, *args, **kwargs):
        self.warning_init_function(self.redo_song_generate.__name__, 'MGGenerate')

    # ---------------------------------------- MGLogistic ----------------------------------------

    def create_token(self, *args, **kwargs):
        self.warning_init_function(self.create_token.__name__, 'MGLogistic')

    def delete_token(self, *args, **kwargs):
        self.warning_init_function(self.delete_token.__name__, 'MGLogistic')

    def get_new_i(self):
        self.warning_init_function(self.get_new_i.__name__, 'MGLogistic')

    def get_new_save_midis_path_i(self, *args, **kwargs):
        self.warning_init_function(self.get_new_save_midis_path_i.__name__, 'MGLogistic')

    def create_token_midis_path(self, *args, **kwargs):
        self.warning_init_function(self.create_token_midis_path.__name__, 'MGLogistic')

    def delete_token_midis_path(self, *args, **kwargs):
        self.warning_init_function(self.delete_token_midis_path.__name__, 'MGLogistic')

    def delete_tokens(self, *args, **kwargs):
        self.warning_init_function(self.delete_tokens.__name__, 'MGLogistic')

    def create_tokens(self, *args, **kwargs):
        self.warning_init_function(self.create_tokens.__name__, 'MGLogistic')

    # ---------------------------------------- MGModel ----------------------------------------

    def new_nn_model(self, *args, **kwargs):
        self.warning_init_function(self.new_nn_model.__name__, 'MGModel')

    def recreate_model(self, *args, **kwargs):
        self.warning_init_function(self.recreate_model.__name__, 'MGModel')

    def load_weights(self, *args, **kwargs):
        self.warning_init_function(self.load_weights.__name__, 'MGModel')

    def print_model(self, *args, **kwargs):
        self.warning_init_function(self.print_model.__name__, 'MGModel')

    def save_model(self, *args, **kwargs):
        self.warning_init_function(self.save_model.__name__, 'MGModel')

    def print_weights(self, *args, **kwargs):
        self.warning_init_function(self.print_weights.__name__, 'MGModel')

    # ---------------------------------------- MGReplicate ----------------------------------------

    def replicate_from_data(self, *args, **kwargs):
        self.warning_init_function(self.replicate_from_data.__name__, 'MGReplicate')

    def replicate_fill(self, *args, **kwargs):
        self.warning_init_function(self.replicate_fill.__name__, 'MGReplicate')

    def redo_song_replicate(self, *args, **kwargs):
        self.warning_init_function(self.redo_song_replicate.__name__, 'MGReplicate')

    # ---------------------------------------- MGTrain ----------------------------------------

    def train(self, *args, **kwargs):
        self.warning_init_function(self.train.__name__, 'MGTrain')

    def evaluate(self, *args, **kwargs):
        self.warning_init_function(self.evaluate.__name__, 'MGTrain')

    def test_on_batch(self, *args, **kwargs):
        self.warning_init_function(self.test_on_batch.__name__, 'MGTrain')

    def predict_on_batch(self, *args, **kwargs):
        self.warning_init_function(self.predict_on_batch.__name__, 'MGTrain')

    def compare_test_predict_on_batch(self, *args, **kwargs):
        self.warning_init_function(self.compare_test_predict_on_batch.__name__, 'MGTrain')




