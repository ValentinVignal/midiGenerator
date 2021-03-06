import argparse

from src import GlobalVariables as g

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
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.HPSearch, ArgType.NScriptsBO]:
            self.add_create_model_args(argtype)
        # Load Model
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.Generate]:
            self.add_load_model_args(argtype)
        # Load Data
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.HPSearch, ArgType.Generate, ArgType.NScriptsBO]:
            self.add_load_data_args(argtype)
        # Train Model
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.HPSearch, ArgType.NScriptsBO]:
            self.add_train_args(argtype)
        # Evaluate Model
        if argtype in [ArgType.ALL, ArgType.Train]:
            self.add_evaluate_model_args(argtype)
        # Generate
        if argtype in [ArgType.ALL, ArgType.Train, ArgType.Generate]:
            self.add_generation_args(argtype)
        # HP Search
        if argtype in [ArgType.ALL, ArgType.HPSearch, ArgType.NScriptsBO]:
            self.add_hp_search_args(argtype)
        # Clean
        if argtype in [ArgType.ALL, ArgType.Clean]:
            self.add_clean_args(argtype)
        # Zip
        if argtype in [ArgType.ALL, ArgType.Zip]:
            self.add_zip_args(argtype)
        # HPSummary
        if argtype in [ArgType.ALL, ArgType.HPSummary]:
            self.add_hp_summary_args(argtype)
        # NScriptsBO
        if argtype in [ArgType.NScriptsBO]:
            self.add_n_scripts_bo_args(argtype)

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
        elif argtype is ArgType.Clean:
            description = 'To clean the environment'
        elif argtype is ArgType.Zip:
            description = 'To zip the files'
        elif argtype is ArgType.HPSummary:
            'To create the summary and images from checkpoint of bayesian optimization search'
        return description

    @staticmethod
    def get_type(argtype, t=float):
        """

        :param argtype:
        :param t:
        :return:
        """
        if argtype in [ArgType.HPSearch, ArgType.NScriptsBO]:
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
        self.add_argument('--fast-seq', default=False, action='store_true',
                          help='Create a FastSequence instance before training')
        self.add_argument('--memory-seq', default=False, action='store_true',
                          help='Store all the data in memory but for a sequence')
        self.add_argument('--validation', type=float,
                          help='Fraction of the training data to be used as validation data')
        self.add_argument('--predict-offset', type=int,
                          help='The offset of the predicted step')
        self.add_argument('--max-queue-size', type=int,
                          help='Max queue size for the function fit_generator')
        self.add_argument('--workers', type=int,
                          help='Number of workers for fit_generator')

        # ---------- Default values ----------
        self.set_defaults(
            epochs=g.train.epochs,
            batch=g.train.batch,
            noise=g.train.noise,
            validation=g.train.validation,
            predict_offset=g.train.predict_offset,
            max_queue_size=g.train.max_queue_size,
            workers=g.train.workers
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
        self.add_store_true(name='--no-all-step-rhythm', argtype=argtype,
                            help='Not taking all the output steps for rhythm')
        self.add_argument('--l-semitone', type=self.get_type(argtype, float),
                          help='the Lambda for the semitone loss')
        self.add_argument('--l-tone', type=self.get_type(argtype, float),
                          help='the Lambda for the tone loss')
        self.add_argument('--l-tritone', type=self.get_type(argtype, float),
                          help='the Lambda for the tritone loss')
        self.add_argument('--use-binary', default=False, action='store_true',
                          help='To use binary cross entropy to note activation')

        self.set_defaults(
            loss_name=g.loss.loss_name,
        )

        if argtype not in [ArgType.HPSearch, ArgType.NScriptsBO]:
            self.set_defaults(
                l_scale=g.loss.l_scale,
                l_rhythm=g.loss.l_rhythm,
                l_semitone=g.loss.l_semitone,
                l_tone=g.loss.l_tone,
                l_tritone=g.loss.l_tritone
            )
        else:
            self.set_defaults(
                l_scale='0:4',
                l_rhythm='0:4',
                no_all_step_rhythm='False',
                l_semitone='0:5',
                l_tone='0:5',
                l_tritone='0:5'
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
        self.add_argument('--no-transposed', default=False, action='store_true',
                          help='To not use a transposed dataset')
        # ---------- Model Id and Load ----------
        if argtype in [ArgType.ALL, ArgType.Train]:
            self.add_argument('-m', '--model-id', type=str, default='',
                              help='The model id modelName,modelParam,nbSteps')
        elif argtype in [ArgType.HPSearch, ArgType.NScriptsBO]:
            self.add_argument('--model-name', type=str, default='',
                              help='The model name')
            self.add_argument('--model-param', type=str, default='',
                              help='the model param (json file)')
            self.add_argument('--nb-steps', type=str, default='',
                              help='Nb step to train on')

        # -------------------- Architecture --------------------
        self.add_argument('--dropout-d', type=self.get_type(argtype, float),
                          help='Value of the dropout for dense layers')
        self.add_argument('--dropout-c', type=self.get_type(argtype, float),
                          help='Value of the dropout for the convolutional layers')
        self.add_argument('--dropout-r', type=self.get_type(argtype, float),
                          help='Value of the dropout for the RNN layers')
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
        self.add_store_true(name='--sah', argtype=argtype,
                            help='use an attention head after the first layer of LSTM')
        self.add_store_true(name='--no-rpoe', argtype=argtype,
                            help='To disable to give previous poe to next step')
        self.add_store_true(name='--no-prior-expert', argtype=argtype,
                            help='No prior expert in PoE')

        self.set_defaults(
            epochs_drop=g.nn.epochs_drop,
            decay_drop=g.nn.decay_drop,
            loss_name=g.loss.loss_name,
            name='name',
            work_on=g.mg.work_on,
        )

        if argtype not in [ArgType.HPSearch, ArgType.NScriptsBO]:
            self.set_defaults(
                lr=g.nn.lr,
                optimizer='adam',
                decay=g.nn.decay,
                dropout_d=g.nn.dropout_r,
                dropout_c=g.nn.dropout_c,
                dropout_r=g.nn.dropout_r,
                all_sequence=g.nn.all_sequence,
                lstm_state=g.nn.lstm_state,
                no_sampling=not g.nn.sampling,
                no_kld=not g.nn.kld,
                kld_annealing_start=g.nn.kld_annealing_start,
                kld_annealing_stop=g.nn.kld_annealing_stop,
                no_kld_sum=not g.nn.kld_sum,
                sah=g.nn.sah,
                no_rpoe=not g.nn.rpoe,
                no_prior_expert=not g.nn.prior_expert
            )
        else:
            self.set_defaults(
                lr='1:4',
                optimizer='adam',
                decay='0:3',
                dropout_d='0.01:0.4',
                dropout_c='0.01:0.4',
                dropout_r='0.01:0.4',
                all_sequence='False',
                lstm_state='False',
                no_sampling='False',
                no_kld='False',
                kld_annealing_start='0:0.5',
                kld_annealing_stop='0.5:1',
                no_kld_sum='False',
                sah='False',
                no_rpoe='False,True',
                no_prior_expert='False,True'
            )

    def add_generation_args(self, artype=ArgType.ALL):
        """

        :param self:
        :param artype:
        :return:
        """
        self.add_argument('--generate', default=False, action='store_true',
                          help='Generation after training')
        self.add_argument('--generate-noise', default=False, action='store_true',
                          help='Generation after training from noise')
        self.add_argument('--replicate', default=False, action='store_true',
                          help='Replication after training')
        self.add_argument('--generate-fill', default=False, action='store_true',
                          help='Fill the missing instrument')
        self.add_argument('--replicate-fill', default=False, action='store_true',
                          help='Fill the missing instrument')
        self.add_argument('--compare-generation', default=False, action='store_true',
                          help='Compare generation after training')
        self.add_argument('--song-number', default=-1, type=int,
                          help='Number of the song in the dataset')
        self.add_argument('--redo-generate', default=False, action='store_true',
                          help='Redo a song by changing one by one the instruments (generate method)')
        self.add_argument('--redo-replicate', default=False, action='store_true',
                          help='Redo a song by changing one by one the instruments (replicate method)')
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
        self.add_argument('--no-train', default=False, action='store_true',
                          help='Disable the training')

        self.add_argument('-l', '--load', type=str, default='',
                          help='The name of the train model to load')
        self.add_argument('--no-save', default=False, action='store_true',
                          help='Won t save the model')
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
        self.add_argument('-d', '--data', type=str, default='BachChoraleBig',
                          help='The name of the data')
        self.add_argument('--data-test', type=str, default='',
                          help='The name of the test data')
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
        self.add_argument('--no-transpose', action='store_true', default=False,
                          help='To disable the transposition to C major')

        return self

    def add_hp_search_args(self, argtype=ArgType.ALL):
        """

        :param argtype:
        :return:
        """
        self.add_argument('--n-calls', type=int, default=20,
                          help='Number of point for the bayesian search')
        self.add_argument('--nb-inter-threads', type=int, default=1,
                          help='Number of inter thread in tensorflow')
        self.add_argument('--nb-intra-threads', type=int, default=1,
                          help='Number of intra thread in tensorflow')
        self.add_argument('--from-checkpoint', default=None,
                          help='To continue the optimization from a checkpoint')
        self.add_argument('--bo-name', default=None,
                          help='Name of the bayesian optimization')
        self.add_argument('--in-place', default=False, action='store_true',
                          help='If continuing from a checkpoint and this arg set to True, '
                               'then it is using the same folder to save the results')

    def add_clean_args(self, argtype=ArgType.ALL):
        """

        :param argtype:
        :return:
        """
        self.add_argument('--midi', default=False, action='store_true',
                          help='Don t delete generated_midis folder')
        self.add_argument('--hp', default=False, action='store_true',
                          help='Don t delete hp_search folder')
        self.add_argument('--model', default=False, action='store_true',
                          help='Don t delete saved_models folder')
        self.add_argument('--tensorboard', default=False, action='store_true',
                          help='Don t delete tensorboard folder')
        self.add_argument('--temp', default=False, action='store_true',
                          help='Don t delete temp folder')
        self.add_argument('--data-temp', default=False, action='store_true',
                          help='Don t delete temp folder in dataset folder')
        self.add_argument('--zip', default=False, action='store_true',
                          help='Delete the file my_zip.zip')

    def add_zip_args(self, argtype=ArgType.ALL):
        """

        :param argtype:
        :return:
        """
        self.add_argument('--midi', default=False, action='store_true',
                          help='To include generated_midis folder')
        self.add_argument('--no-midi', default=False, action='store_true',
                          help='To exclude generated_midis folder')
        self.add_argument('--hp', default=False, action='store_true',
                          help='To include hp_search folder')
        self.add_argument('--no-hp', default=False, action='store_true',
                          help='To exclude hp_search folder')
        self.add_argument('--model', default=False, action='store_true',
                          help='To include saved_models folder')
        self.add_argument('--no-model', default=False, action='store_true',
                          help='To exclude saved_models folder')

    def add_hp_summary_args(self, argtype=ArgType.ALL):
        """

        :param argtype:
        :return:
        """
        self.add_argument('folder', type=int,
                          help='Number of the bayesian optimization folder')

    def add_n_scripts_bo_args(self, argtype=ArgType.NScriptsBO):
        """

        :param argtype:
        :return:
        """
        self.add_argument('nscripts', type=int, default=1,
                          help='Number of time to run the script')
