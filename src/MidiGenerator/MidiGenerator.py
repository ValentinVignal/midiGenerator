import src.global_variables as g
from .MGPredict import MGPredict
from .MGData import MGData
from .MGTrain import MGTrain
from .MGModel import MGModel
from .MGLogistic import MGLogistic
from .MGInit import MGInit


class MidiGenerator(MGPredict, MGData, MGTrain, MGModel, MGLogistic, MGInit):
    """

    """

    def __init__(self, *args, **kwargs):
        MGInit.__init__(self, *args, **kwargs)

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

