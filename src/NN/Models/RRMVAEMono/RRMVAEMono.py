import tensorflow as tf

from src import GlobalVariables as g
import src.NN.layers as mlayers
from src.NN.Models.KerasModel import KerasModel
import src.NN.shapes.convolution as s_conv
import src.NN.shapes.time as s_time
from src.eval_string import eval_object
from src.NN import Loss
from src import dictionaries

layers = tf.keras.layers
K = tf.keras.backend


class RRMVAEMono(KerasModel):

    def __init__(self, *args, replicate=False, scale=False, **kwargs):
        super(RRMVAEMono, self).__init__(*args, **kwargs)
        self.replicate = replicate
        self.scale = scale

    def generate(self, x, *args, **kwargs):
        """
        Same as predict but without messing output

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        y = self.predict(x=x, *args, **kwargs)
        # y = List(nb_instruments) + [all instruments]
        if self.scale:
            # Remove the all_outputs tensor at the end of the list
            return y[:-1]
        else:
            # Return all the outputs
            return y


def create(
        # Mandatory arguments
        input_param, model_param, nb_steps, step_length, optimizer, model_options={}, loss_options={},
        # Hidden arguments
        replicate=False, music=False
):
    """

    # Mandatory arguments
    :param loss_options:
    :param input_param: {
                            nb_instruments,
                            input_size
                        }
                        (Comes from the dataset)
    :param model_param: (Comes from the .json)
    :param nb_steps: (Comes from the user)
    :param step_length: (Comes from the user)
    :param optimizer:
    :param model_options: (Comes from the user)

    # Hidden_arguments
    :param replicate: Bool: Indicates if this is a model to replicate the input
    :param music: Bool, indicate if we want to use the concatenated outputs at the end

    :return: the neural network:
    """

    model_options_default = dict(
        dropout_d=g.nn.dropout_d,
        dropout_c=g.nn.dropout_c,
        dropout_r=g.nn.dropout_r,
        sampling=g.nn.sampling,
        kld=g.nn.kld,
        kld_annealing_start=g.nn.kld_annealing_start,
        kld_annealing_stop=g.nn.kld_annealing_stop,
        kld_sum=g.nn.kld_sum,
        sah=g.nn.sah,
        rpoe=g.nn.rpoe,
        prior_expert=g.nn.prior_expert
    )
    dictionaries.set_default(model_options, model_options_default)

    loss_options_default = dict(
        loss_name='mono',
        l_scale=g.loss.l_scale,
        l_rhythm=g.loss.l_rhythm,
        take_all_step_rhythm=g.loss.take_all_step_rhythm,
        l_semiton=g.loss.l_semitone,
        l_tone=g.loss.l_tone,
        l_tritone=g.loss.l_tritone,
        use_binary=g.loss.use_binary
    )
    dictionaries.set_default(loss_options, loss_options_default)

    print('Definition of the graph ...')

    # --------- End model options ----------

    # --------- Variables ----------

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    model_param = eval_object(model_param, env=env)

    midi_shape = (nb_steps, step_length, input_size, 1)  # (batch, nb_steps, step_length, input_size, 2)
    mask_shape = (nb_instruments, nb_steps)

    dense_note_decoder = model_param['dense_note'][::-1] + [input_size]

    # --------- End Variables ----------

    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, step_length, input_size, 2)]
    input_mask = tf.keras.Input(mask_shape)  # (batch, nb_instruments, nb_steps)

    # It's a mono model, let's reshape
    inputs_midi_reshaped = [
        layers.Reshape((nb_steps, step_length, input_size))(inp) for inp in inputs_midi
    ]  # List(nb_instruments)[(nb_steps, step_length, input_size)]

    # ------------------------------ Encoding Notes ------------------------------

    encoders_note = [mlayers.dense.DenseCoder(
        size_list=model_param['dense_note'],
        dropout=model_options['dropout_d']
    ) for _ in range(nb_instruments)]

    note_encoded = mlayers.wrapper.func.apply_different_on_list(
        layers=encoders_note
    )(inputs_midi_reshaped)     # List(nb_instrument)[(nb_steps, step_length, input_size)]

    # TODO: Use bidirectional
    lstm_encoder_note = [
        mlayers.rnn.LstmRNN(
            size_list=model_param['lstm_note'],
            dropout=model_options['dropout_r'],
            bidirectionnal=True
        ) for _ in range(nb_instruments)]

    notes_encoded_inst_step = [
        mlayers.shapes.Unstack(axis=0)(inp) for inp in note_encoded
    ]  # List(nb_instruments, nb_steps)[(step_length, input_size)]
    notes_encoded_step_inst = mlayers.shapes.transpose_list(
        l=notes_encoded_inst_step,
        axes=(1, 0)
    )       # List(nb_steps, nb_instruments)[(step_length, input_size)]

    notes_lstm_encoded_step_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_on_list(layers=lstm_encoder_note)
    )(notes_encoded_step_inst)      # List(nb_steps, nb_instruments)[(size,)]

    notes_lstm_encoded_inst_step = mlayers.shapes.transpose_list(
        l=notes_lstm_encoded_step_inst,
        axes=(1, 0)
    )       # List(nb_instruments, nb_steps)[(size,)]

    notes_lstm_encoded_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.shapes.Stack(axis=0)
    )(notes_lstm_encoded_inst_step)         # List(nb_instruments)[(nb_steps, size)]

    # ------------------------------ Encoding Measure ------------------------------

    encoders_measure = [
        mlayers.dense.DenseCoder(
            size_list=model_param['dense_measure'],
            dropout=model_options['dropout_d']
        ) for _ in range(nb_instruments)]

    measure_encoded_inst = mlayers.wrapper.func.apply_different_on_list(
        layers=encoders_measure
    )(notes_lstm_encoded_inst)      # List(nb_instruments)[(nb_steps, size)]

    # -------------------- Product of Expert --------------------

    latent_size = model_param['dense_measure'][-1]
    denses_for_mean = [mlayers.dense.DenseForMean(units=latent_size) for inst in range(nb_instruments)]
    denses_for_std = [mlayers.dense.DenseForSTD(units=latent_size) for inst in range(nb_instruments)]

    means_inst = mlayers.wrapper.func.apply_different_on_list(
        layers=denses_for_mean
    )(measure_encoded_inst)     # List(nb_instruments)[(nb_steps, size)]
    stds_inst = mlayers.wrapper.func.apply_different_on_list(
        layers=denses_for_std
    )(measure_encoded_inst)     # List(nb_instruments)[(nb_steps, size)]

    means = mlayers.shapes.Stack(axis=0)(means_inst)  # (batch, nb_instruments, nb_steps, size)
    stds = mlayers.shapes.Stack(axis= 0)(stds_inst)  # (batch, nb_instruments, nb_steps, size)

    if model_options['prior_expert']:
        means, stds, mask = mlayers.vae.AddPriorExper()([means, stds, input_mask])
    else:
        mask = input_mask

    if model_options['rpoe']:
        poe = mlayers.vae.RPoeMask(axis=0)([means, stds, mask])  # List(2)[(batch, nb_steps, size)]
    else:
        poe = mlayers.vae.ProductOfExpertMask(axis=0)([means, stds, mask])  # List(2)[(batch, nb_steps, size)]
    if model_options['kld']:
        sum_axis = 0 if model_options['kld_sum'] else None
        kld = mlayers.vae.KLDAnnealing(
            sum_axis=sum_axis,
            epoch_start=model_options['kld_annealing_start'],
            epoch_stop=model_options['kld_annealing_stop']
        )(poe)
    if model_options['sampling']:
        samples = mlayers.vae.SampleGaussian()(poe)
    else:
        samples = layers.Concatenate(axis=-1)(poe)  # (batch, nb_steps, size)

    # ------------------------------ RNN ------------------------------

    rnn_output = mlayers.rnn.LstmRNN(
        size_list=model_param['lstm_measure'],
        return_sequences=replicate,
        use_sah=model_options['sah'],
        dropout=model_options['dropout_r']
    )(samples)  # (batch, steps, size) if replicate else (batch, size)

    if replicate:
        rnn_output_steps = mlayers.shapes.Unstack(axis=0)(rnn_output)  # List(nb_steps)[(batch, size)]
    else:
        rnn_output_steps = [rnn_output]  # List(1)[(batch, size)]
    # Now either case, we have rnn_output_steps : List(nb_steps)[(batch, size)]

    # ------------------------------ Decoding Measure ------------------------------

    stacked_rnn_output = mlayers.shapes.Stack(axis=0)(rnn_output_steps)     # (batch, nb_steps, size)

    measure_decoders = [
        mlayers.dense.DenseCoder(
            size_list=model_param['dense_measure'][::-1],
            dropout=model_options['dropout_d']
        ) for _ in range(nb_instruments)]

    decoded_measure = mlayers.wrapper.func.apply_different_layers(
        layers=measure_decoders
    )(stacked_rnn_output)       # List(nb_instruments)[(batch, nb_steps, size)]

    # ------------------------------ Decoding Note ------------------------------

    decoded_measure_inst_step = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.shapes.Unstack(axis=0)
    )(decoded_measure)      # List(nb_instruments, nb_steps)[(batch, size)]
    decoded_measure_step_inst = mlayers.shapes.transpose_list(
        l=decoded_measure_inst_step,
        axes=(1, 0)
    )       # List(nb_steps, nb_instruments)[(batch, size)]

    measure_decoders_lstm = [
        mlayers.rnn.MultiLSTMGen(
            nb_steps=step_length,
            size_list=model_param['lstm_note'][::-1],
            dropout=model_options['dropout_r'],
            bidirectional=True
        ) for _ in range(nb_instruments)]

    decoded_notes_step_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_on_list(
            layers=measure_decoders_lstm
        )
    )(decoded_measure_step_inst)        # List(nb_steps, nb_instruments)[(batch, step_length, size)]

    decoders_notes = [mlayers.dense.DenseCoder(
        size_list=dense_note_decoder,
        dropout=model_options['dropout_d']
        ) for _ in range(nb_instruments)]

    decoded_notes_step_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_on_list(
            layers=decoders_notes
        )
    )(decoded_notes_step_inst)      # List(nb_steps, nb_instruments)[(batch, step_length, size)]

    # Last layer
    if loss_options['use_binary']:
        last_mono = [mlayers.last.LastInstMonoBinary(softmax_axis=-2) for inst in range(nb_instruments)]
    else:
        # last_mono = [mlayers.last.LastInstMono(softmax_axis=-2) for inst in range(nb_instruments)]
        last_mono = [mlayers.dense.DenseSameShape(activation_layer=layers.Softmax(axis=-1)) for _ in range(nb_instruments)]
    outputs_step_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_on_list(layers=last_mono)
    )(decoded_notes_step_inst)  # List(nb_steps, nb_instruments)[(batch, step_length, size)]

    # Add the channels
    outputs_step_inst_withchannels = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_same_on_list(
            layer=mlayers.shapes.ExpandDims(axis=-1)
        )
    )(outputs_step_inst)      # List(nb_steps, nb_instruments)[(batch, step_length, size, channels=1)]    (Mono)
    outputs_inst_steps = mlayers.shapes.transpose_list(
        outputs_step_inst_withchannels,
        axes=(1, 0))  # List(nb_instruments, nb_steps)[batch, step_length, size, channels)]

    outputs = [mlayers.shapes.Stack(axis=0)(o) for o in
               outputs_inst_steps]  # List(nb_instruments)[(batch, nb_steps, step_length, size, channel=1)]
    outputs = [layers.Layer(name=f'Output_{inst}')(outputs[inst]) for inst in range(nb_instruments)]

    if music:
        all_outputs = mlayers.shapes.Stack(name='All_outputs', axis=0)(outputs)
        # all_outputs: (batch, nb_instruments, nb_steps, step_size, input_size, channels=1)
        outputs = outputs + [all_outputs]
        # harmony_loss = Loss.cost.harmony(**loss_options)(all_outputs[:, :, :, :, :-1, 0])
        harmony = mlayers.music.Harmony(
            l_semitone=loss_options['l_semitone'],
            l_tone=loss_options['l_tone'],
            l_tritone=loss_options['l_tritone'],
            mono=True
        )(all_outputs)

    model = RRMVAEMono(inputs=inputs_midi + [input_mask], outputs=outputs,
                       scale=music, replicate=replicate)

    # ------------------ Losses -----------------
    # Define losses dict for outputs and metric
    losses = {}
    metrics = {}
    for inst in range(nb_instruments):
        losses[f'Output_{inst}'] = Loss.from_names[loss_options['loss_name']](
            **loss_options
        )
        # metrics[f'Output_{inst}'] = Loss.metrics.acc_mono()
        if loss_options['use_binary']:
            metrics[f'Output_{inst}'] = [Loss.metrics.acc_mono_bin(), Loss.metrics.acc_mono_cat()]
        else:
            metrics[f'Output_{inst}'] = Loss.metrics.acc_mono()
    if music:
        losses['All_outputs'] = Loss.scale(**loss_options, mono=True)

    # Define kld
    if model_options['kld']:
        model.add_loss(kld)

    # harmony
    if music:
        model.add_loss(harmony)

    # ------------------ Metrics -----------------

    # -------------------- Callbacks --------------------
    callbacks = []
    # ------------------------------ Compile ------------------------------

    model.compile(loss=losses,
                  optimizer=optimizer,
                  metrics=metrics)
    if model_options['kld']:
        model.add_metric(kld, name='kld', aggregation='mean')
    if music:
        model.add_metric(harmony, name='harmony', aggregation='mean')

    return dict(
        model=model,
        callbacks=callbacks
    )


def get_create(replicate=False, music=False):
    """

    :param replicate:
    :param music:
    :return:
    """

    def _create(*args, **kwargs):
        return create(*args, replicate=replicate, music=music, **kwargs)

    return _create
