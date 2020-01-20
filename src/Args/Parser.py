import argparse

from src import global_variables as g

from .ArgType import ArgType


class Parser(argparse.ArgumentParser):
    """
    custom argument parser for MidiGenerator scripts
    """

    def __init__(self, *args, argtype=None, **kwargs):
        """

        :param args:
        :param argtype:
        :param kwargs:
        """
        self.argtype = argtype
        if 'description' not in kwargs and argtype is not None:
            kwargs['description'] = self.description_msg(argtype)
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super(Parser, self).__init__(*args, **kwargs)
        if argtype is not None:
            self.do_init(argtype)

    def do_init(self, argtype):
        """

        :param argtype:
        :return:
        """
        self.add_execution_type_args(argtype)
        # ---------- Compute the dataset ----------
        if argtype in [ArgType.ALL, ArgType.ComputeData, ArgType.CheckData]:
            self.add_compute_data_args(argtype)
        # ---------- Model ----------
        # Creation Model
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.HPSearch]:
            self.add_create_model_args(argtype)
        # Load Model
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.Generate]:
            self.add_load_model_args(argtype)
        # Load Data
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.HPSearch, ArgType.Generate]:
            self.add_load_data_args(argtype)
        # Train Model
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.HPSearch]:
            self.add_train_args(argtype)
        # Evaluate Model
        if argtype in [ArgType.ALL, ArgType.Train]:
            self.add_evaluate_model_args(argtype)
        # Generate
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.Generate]:
            self.add_generation_args(argtype)
        # HP Search
        if argtype in [ArgType.ALL, ArgType.HPSearch]:
            self.add_hp_search_args(argtype)

    @staticmethod
    def description_msg(argtype=None):
        """

        :param argtype:
        :return: The description message for the ArgumentParser
        """
        description = 'Default description'
        if argtype is ArgType.Train:
            description = 'To train the model'
        elif argtype is ArgType.HPSearch:
            description = 'To find the best Hyper Parameters'
        elif argtype is ArgType.Generate:
            description = 'To Generate music from a trained model'
        elif argtype is ArgType.ComputeData:
            description = 'To compute the data'
        elif argtype is ArgType.CheckData:
            description = 'To Check the data'
        return description

    @staticmethod
    def get_type(argtype, t=float):
        """

        :param argtype:
        :param t:
        :return:
        """
        if argtype in [ArgType.HPSearch]:
            return str
        else:
            return t

    def add_store_true(self, name, argtype=ArgType.ALL, help=''):
        """
        Add to the parser a store true action
        :param self:
        :param name:
        :param argtype:
        :param help:
        :return:
        """
        if argtype not in [ArgType.HPSearch]:
            self.add_argument(name, action='store_true',
                              help=help)
        else:
            self.add_argument(name, type=str,
                              help=help)

    def add_train_args(self, argtype=ArgType.ALL):
        """

        :param self:
        :param argtype:
        :return:
        """
        self.add_argument('-e', '--epochs', type=int,
                          help='number of epochs to train')
        self.add_argument('-b', '--batch', type=int,
                          help='The number of the batches')
        self.add_argument('--noise', type=float,
                          help='If not 0, add noise to the input for training')
        self.add_argument('--seq2np', default=False, action='store_true',
                          help='For small dataset, store all the data in a numpy array to train faster')
        self.add_argument('--validation', type=float,
                          help='Fraction of the training data to be used as validation data')

        # ---------- Default values ----------
        self.set_defaults(
            epochs=g.epochs,
            batch=g.batch,
            noise=g.noise,
            validation=g.validation
        )

    def add_evaluate_model_args(self, argtype=ArgType.ALL):
        """

        :param self:
        :param argtype:
        :return:
        """
        self.add_argument('--evaluate', default=False, action='store_true',
                          help='Evaluate the model after the training')
        self.add_argument('--check-batch', type=int, default=-1,
                          help='Batch to check')

    def add_loss_args(self, argtype=ArgType.ALL):
        """

        :param argtyp:
        :return:
        """
        self.add_argument('--loss-name', type=str,
                          help='Name of the loss')
        self.add_argument('--l-scale', type=self.get_type(argtype, float),
                          help='Lambda for scale loss')
        self.add_argument('--l-rhythm', type=self.get_type(argtype, float),
                          help='Lambda for the rhythm loss')
        self.add_argument('--l-scale-cost', type=self.get_type(argtype, float),
                          help='The cost for an out of scale note')
        self.add_argument('--l-rhythm-cost', type=self.get_type(argtype, float),
                          help='The cost for an out of rhythm note')
        self.add_store_true(name='--no-all-step-rhythm', argtype=argtype,
                            help='Not taking all the output steps for rhythm')

        self.set_defaults(
            loss_name=g.loss_name,
        )

        if argtype is not ArgType.HPSearch:
            self.set_defaults(
                l_scale=g.l_scale,
                l_rhythm=g.l_rhythm,
                l_scale_cost=g.l_scale_cost,
                l_rhythm_cost=g.l_rhythm_cost
            )
        else:
            self.set_defaults(
                l_scale='0:4',
                l_rhythm='0:4',
                l_scale_cost='0:4',
                l_rhythm_cost='0:4',
                no_all_step_rhythm='False,True'
            )

    def add_create_model_args(self, argtype=ArgType.ALL):
        """

        :param self:
        :param argtype:
        :return:
        """
        # -------------------- Training parameters --------------------
        self.add_argument('--lr', type=self.get_type(argtype, float),
                          help='learning rate')
        self.add_argument('-o', '--optimizer', type=str,
                          help='Name of the optimizer')
        self.add_argument('--decay', type=self.get_type(argtype, t=float),
                          help='Learning Rate Decay')
        self.add_argument('--epochs-drop', type=float,
                          help='how long before a complete drop (decay)')
        self.add_argument('--decay-drop', type=float,
                          help='0 < decay_drop < 1, every epochs_drop, lr will be multiply by decay_drop')
        self.add_loss_args(argtype=argtype)
        # -------------------- Model Type --------------------
        self.add_argument('-n', '--name', type=str,
                          help='Name given to the model')
        self.add_argument('--work-on', type=str,
                          help='note, beat or measure')
        self.add_argument('--mono', default=False, action='store_true',
                          help='To work with monophonic instruments')
        # ---------- Model Id and Load ----------
        if argtype in [ArgType.ALL, ArgType.Train]:
            self.add_argument('-m', '--model-id', type=str, default='',
                              help='The model id modelName,modelParam,nbSteps')
        elif argtype is ArgType.HPSearch:
            self.add_argument('--model-name', type=str, default='',
                              help='The model name')
            self.add_argument('--model-param', type=str, default='',
                              help='the model param (json file)')
            self.add_argument('--nb-steps', type=str, default='',
                              help='Nb step to train on')

        # -------------------- Architecture --------------------
        self.add_argument('--dropout', type=self.get_type(argtype, float),
                          help='Value of the dropout')
        self.add_store_true(name='--all-sequence', argtype=argtype,
                            help='Use or not all the sequence in the RNN layer')
        self.add_store_true(name='--lstm-state', argtype=argtype,
                            help='Use or not all the sequence in the RNN layer')
        self.add_store_true(name='--no-sampling', argtype=argtype,
                            help='Gaussian Sampling')
        self.add_store_true(name='--no-kld', argtype=argtype,
                            help='No KL Divergence')
        self.add_argument('--kld-annealing-start', type=self.get_type(argtype, float),
                          help='Start of the annealing of the kld')
        self.add_argument('--kld-annealing-stop', type=self.get_type(argtype, float),
                          help='Stop of the annealing of the kld')
        self.add_store_true(name='--no-kld-sum', argtype=argtype,
                            help='To not sum through time for the KLD')

        self.set_defaults(
            epochs_drop=g.epochs_drop,
            decay_drop=g.decay_drop,
            loss_name=g.loss_name,
            name='name',
            work_on=g.work_on,
        )

        if argtype is not ArgType.HPSearch:
            self.set_defaults(
                lr=g.lr,
                optimizer='adam',
                decay=g.decay,
                dropout=g.dropout,
                all_sequence=g.all_sequence,
                lstm_state=g.lstm_state,
                no_sampling=False,
                no_kld=False,
                kld_annealing_start=g.kld_annealing_start,
                kld_annealing_stop=g.kld_annealing_stop,
                no_kld_sum=False
            )
        else:
            self.set_defaults(
                lr='1:4',
                optimizer='adam',
                decay='0.01:1',
                dropout='0.1:0.3',
                all_sequence='False',
                lstm_state='False',
                no_sampling='False',
                no_kld='False',
                kld_annealing_start='0:0.5',
                kld_annealing_stop='0.5:1',
                no_kld_sum='False'
            )

    def add_generation_args(self, artype=ArgType.ALL):
        """

        :param self:
        :param artype:
        :return:
        """
        self.add_argument('--compare-generation', default=False, action='store_true',
                          help='Compare generation after training')
        self.add_argument('--generate', default=False, action='store_true',
                          help='Generation after training')
        self.add_argument('--replicate', default=False, action='store_true',
                          help='Replication after training')
        self.add_argument('--generate-fill', default=False, action='store_true',
                          help='Fill the missing instrument')
        self.add_argument('--replicate-fill', default=False, action='store_true',
                          help='Fill the missing instrument')
        self.add_argument('--no-duration', action='store_true', default=False,
                          help='Generate only shortest notes possible')

        self.add_argument('--nb-seeds', default=4,
                          help='number of seeds or the path to the folder with the seeds')

    def add_load_model_args(self, argtype=ArgType.ALL):
        """

        :param self:
        :param argtype:
        :return:
        """

        self.add_argument('-l', '--load', type=str, default='',
                          help='The name of the train model to load')
        return self

    def add_execution_type_args(self, argtype=ArgType.ALL):
        """

        :param self:
        :param argtype:
        :return:
        """
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.Generate, argtype.HPSearch]:
            self.add_argument('--gpu', type=str, default='0',
                              help='What GPU to use')
            self.add_argument('--no-eager', default=False, action='store_true',
                              help='Disable eager execution')
        self.add_argument('--pc', action='store_true', default=False,
                          help='To work on a small computer with a cpu')
        self.add_argument('--no-pc-arg', action='store_true', default=False,
                          help='To no transform parameters during pc execution')
        self.add_argument('--debug', action='store_true', default=False,
                          help='To set special parameters for a debug')

        return self

    def add_load_data_args(self, argtype=ArgType.ALL):
        """

        :param self:
        :param argtype:
        :return:
        """
        self.add_argument('-d', '--data', type=str, default='lmd_matched_small',
                          help='The name of the data')
        return self

    def add_compute_data_args(self, argtype=ArgType.ALL):
        """

        :param self:
        :param argtype:
        :return:
        """

        self.add_argument('data', type=str, default='',
                          help='The name of the data')
        if argtype in [ArgType.ALL, ArgType.ComputeData]:
            self.add_argument('--length', type=str, default='',
                              help='The length of the data')
        self.add_argument('--notes-range', type=str, default='0:88',
                          help='The length of the data')
        self.add_argument('--instruments', type=str, default='Piano,Trombone',
                          help='The instruments considered (for space in name, put _ instead : Acoustic_Bass)')
        self.add_argument('--bach', action='store_true', default=False,
                          help='To compute the bach data')
        self.add_argument('--mono', action='store_true', default=False,
                          help='To compute the data where there is only one note at the same time')
        if argtype in [ArgType.ALL, ArgType.CheckData]:
            self.add_argument('--images', action='store_true', default=False,
                              help='To also create the pianoroll')

        return self

    def add_hp_search_args(self, argtype=ArgType.ALL):
        """

        :param argtype:
        :return:
        """
        self.add_argument('--n-calls', type=int, default=20,
                          help='Number of point for the bayesian search')
