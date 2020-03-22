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


class RMVAEMono(KerasModel):

    def __init__(self, *args, replicate=False, scale=False, **kwargs):
        super(RMVAEMono, self).__init__(*args, **kwargs)
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

    time_stride = s_time.time_stride(step_length)

    midi_shape = (nb_steps, step_length, input_size, 1)  # (batch, nb_steps, step_length, input_size, 2)
    mask_shape = (nb_instruments, nb_steps)

    # ----- For the decoder -----

    model_param_dec, shape_before_conv_dec = s_conv.compute_model_param_dec(
        input_shape=midi_shape[1:],
        model_param_enc=model_param,
        strides=[(time_stride, 2) for i in range(len(model_param['conv']) - 1)],
        nb_steps_to_1=False
    )  # Compute the values of the filters for
    # transposed convolution and fully connected + the size of the tensor before the flatten layer in the encoder

    shapes_before_pooling = s_conv.compute_shapes_before_pooling(
        input_shape=midi_shape[1:],
        model_param_conv=model_param['conv'],
        strides=(time_stride, 2)
    )  # All the shapes of the tensors before each pooling
    # Put time to 1 for the output:
    # shapes_before_pooling = s_time.time_step_to_x(l=shapes_before_pooling, axis=1, x=1)
    # To use only if there is nb steps in conv

    # --------- End Variables ----------

    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, step_length, input_size, 2)]
    input_mask = tf.keras.Input(mask_shape)  # (batch, nb_instruments, nb_steps)

    inputs_inst_step = [mlayers.shapes.Unstack(axis=0)(input_inst) for input_inst in
                        inputs_midi]  # List(nb_instruments, nb_steps)[(batch, step_length, size, channels)]
    inputs_step_inst = mlayers.shapes.transpose_list(
        inputs_inst_step,
        axes=(1, 0)
    )  # List(nb_steps, nb_instruments)[(batch, step_length, size, channels)]

    # ------------------------------ Encoding ------------------------------

    encoders = [mlayers.coder2D.Encoder2D(
        encoder_param=model_param,
        dropout_c=model_options['dropout_c'],
        dropout_d=model_options['dropout_d'],
        time_stride=time_stride,
        time_distributed=False
    ) for inst in range(nb_instruments)]

    encoded_step_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_on_list(layers=encoders)
    )(inputs_step_inst)  # List(steps, nb_instruments)[(batch, size)]
    # -------------------- Product of Expert --------------------

    latent_size = model_param['dense'][-1]
    denses_for_mean = [mlayers.dense.DenseForMean(units=latent_size) for inst in range(nb_instruments)]
    denses_for_std = [mlayers.dense.DenseForSTD(units=latent_size) for inst in range(nb_instruments)]

    means_step_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_on_list(layers=denses_for_mean)
    )(encoded_step_inst)  # List(steps, nb_instruments)[(batch, size)]
    stds_step_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_on_list(layers=denses_for_std)
    )(encoded_step_inst)  # List(steps, nb_instruments)[(batch, size)]

    means = mlayers.shapes.Stack(axis=(1, 0))(means_step_inst)  # (batch, nb_instruments, nb_steps, size)
    stds = mlayers.shapes.Stack(axis=(1, 0))(stds_step_inst)  # (batch, nb_instruments, nb_steps, size)

    if model_options['prior_expert']:
        means, stds, mask = mlayers.vae.AddPriorExper()([means, stds, input_mask])
    else:
        mask = input_mask

    if model_options['rpoe']:
        poe = mlayers.vae.RPoeMask(axis=0)([means, stds, mask])       # List(2)[(batch, nb_steps, size)]
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
        size_list=model_param['lstm'],
        return_sequence=replicate,
        use_sah=model_options['sah'],
        dropout=model_options['dropout_r']
    )(samples)  # (batch, steps, size) if replicate else (batch, size)

    if replicate:
        rnn_output_steps = mlayers.shapes.Unstack(axis=0)(rnn_output)     # List(nb_steps)[(batch, size)]
    else:
        rnn_output_steps = [rnn_output]  # List(1)[(batch, size)]
    # Now either case, we have rnn_output_steps : List(nb_steps)[(batch, size)]

    # ------------------------------ Decoding ------------------------------

    decoders = [mlayers.coder2D.Decoder2D(
        decoder_param=model_param_dec,
        shape_before_conv=shape_before_conv_dec,
        dropout_c=model_options['dropout_c'],
        dropout_d=model_options['dropout_d'],
        time_stride=time_stride,
        shapes_after_upsize=shapes_before_pooling,
        last_activation='tanh'
    ) for _ in range(nb_instruments)]
    decoded_steps_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_layers(layers=decoders)
    )(rnn_output_steps)     # List(nb_steps, nb_instruments)[(batch, step_length, size, channels)]

    if loss_options['use_binary']:
        last_mono = [mlayers.last.LastInstMonoBinary(softmax_axis=-2) for inst in range(nb_instruments)]
    else:
        last_mono = [mlayers.last.LastInstMono(softmax_axis=-2) for inst in range(nb_instruments)]
    outputs_steps_inst = mlayers.wrapper.func.apply_same_on_list(
        layer=mlayers.wrapper.func.apply_different_on_list(layers=last_mono)
    )(decoded_steps_inst)       # List(nb_steps, nb_instruments)[(batch, step_length, size, channels)]
    outputs_inst_steps = mlayers.shapes.transpose_list(
        outputs_steps_inst,
        axes=(1, 0))        # List(nb_instruments, nb_steps)[batch, step_length, size, channels)]

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

    model = RMVAEMono(inputs=inputs_midi + [input_mask], outputs=outputs,
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


