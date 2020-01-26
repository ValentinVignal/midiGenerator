from src import GlobalVariables as g
from .MGGenerate import MGGenerate
from .MGData import MGData
from .MGTrain import MGTrain
from .MGModel import MGModel
from .MGLogistic import MGLogistic
from .MGInit import MGInit
from .MGComputeGeneration import MGComputeGeneration
from .MGReplicate import MGReplicate


class MidiGenerator(MGGenerate, MGReplicate, MGComputeGeneration, MGData, MGTrain, MGModel, MGLogistic, MGInit):
    """

    """

    def __init__(self, *args, **kwargs):
        MGInit.__init__(self, *args, **kwargs)

    # --------------------------------------------------
    #               Class Methods
    # --------------------------------------------------

    @classmethod
    def from_model(cls, id, name='name', data=None):
        my_model = cls(name=name, data=data)
        my_model.load_model(id=id)
        return my_model

    @classmethod
    def with_new_model(cls, model_infos, name='name', work_on=g.mg.work_on, data=None):
        my_model = cls(name=name, data=data)

        def get_value(key):
            """

            :param key: key in the dictionary "model_infos"
            :return: the value in model_infos or None if it doesn't exist
            """
            value = None if key not in model_infos else model_infos[key]
            return value

        my_model.input_param = model_infos['input_param']
        my_model.model_id = model_infos['model_id']
        my_model.new_nn_model(
            model_id=model_infos['model_id'],
            work_on=work_on,
            opt_param=get_value('opt_param'),
        )
        return my_model

    @classmethod
    def with_model(cls, id, with_weights=True):
        my_model = cls()
        my_model.recreate_model(id=id, with_weigths=with_weights)
        return my_model

    def __del__(self, *args, **kwargs):
        MGInit.__del__(self, *args, **kwargs)

