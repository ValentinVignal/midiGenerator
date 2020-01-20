import tensorflow as tf

import src.global_variables as g
import src.NN.layers as mlayers
from src.NN.Models.KerasModel import KerasModel
import src.NN.shapes.convolution as s_conv
import src.NN.shapes.time as s_time
from src.eval_string import eval_object
from src.NN import Callbacks
from src import dictionaries
from src.NN import Loss

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda
K = tf.keras.backend


def create_model(input_param, model_param, nb_steps, step_length, optimizer, model_options={}, loss_options={}):
    """

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
    :return: the neural network:
    """

    # ---------- Model options ----------
    model_options_default = dict(
        dropout=g.dropout,
        sampling=g.sampling,
        kld=g.kld,
        kld_annealing_start=g.kld_annealing_start,
        kld_annealing_stop=g.kld_annealing_stop,
        kld_sum=g.kld_sum
    )
    dictionaries.set_default(model_options, model_options_default)

    loss_options_default = dict(
        loss_name='mono',
        l_scale=g.l_scale,
        l_rhythm=g.l_rhythm,
        l_scale_cost=g.l_scale_cost,
        l_rhythm_cost=g.l_rhythm_cost,
        take_all_step_rhythm=g.take_all_step_rhythm
    )
    dictionaries.set_default(loss_options, loss_options_default)

    dropout = model_options['dropout']

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
    )       # List(nb_steps, nb_instruments)[(batch, step_length, size, channels)]

    # ------------------------------ Encoding ------------------------------

    encoders = [mlayers.coder2D.Encoder2D(
        encoder_param=model_param,
        dropout=dropout,
        time_stride=time_stride,
        time_distributed=False
    ) for inst in range(nb_instruments)]
    encoded_step_inst = mlayers.wrapper.l.ApplySameOnList(
        layer=mlayers.wrapper.l.ApplyDifferentOnList(layers=encoders),
        name='encoders'
    )(inputs_step_inst)       # List(nb_steps, nb_instruments)[(batch, size)]
    # -------------------- Product of Expert --------------------

    latent_size = model_param['dense'][-1]
    denses_for_mean = [mlayers.dense.DenseForMean(units=latent_size) for inst in range(nb_instruments)]
    denses_for_std = [mlayers.dense.DenseForSTD(units=latent_size) for inst in range(nb_instruments)]

    means_step_inst = mlayers.wrapper.l.ApplySameOnList(
        layer=mlayers.wrapper.l.ApplyDifferentOnList(layers=denses_for_mean),
        name='mean_denses'
    )(encoded_step_inst)       # List(steps, nb_instruments)[(batch, size)]
    stds_step_inst = mlayers.wrapper.l.ApplySameOnList(
        layer=mlayers.wrapper.l.ApplyDifferentOnList(layers=denses_for_std),
        name='std_denses'
    )(encoded_step_inst)       # List(steps, nb_instruments)[(batch, size)]

    means = mlayers.shapes.Stack(axis=(1, 0))(means_step_inst)      # (batch, nb_instruments, nb_steps, size)
    stds = mlayers.shapes.Stack(axis=(1, 0))(stds_step_inst)        # (batch, nb_instruments, nb_steps, size)

    poe = mlayers.vae.ProductOfExpertMask(axis=0)([means, stds, input_mask])    # List(2)[(batch, nb_steps, size)]
    if model_options['kld']:
        kld_weight = K.variable(0)
        kld_weight._trainable = False
        sum_axis = 0 if model_options['kld_sum'] else None
        kld = mlayers.vae.KLD(kld_weight, sum_axis=sum_axis)(poe)  # (1,)
    if model_options['sampling']:
        samples = mlayers.vae.SampleGaussian()(poe)
    else:
        samples = layers.Concatenate(axis=-1)(poe)      # (batch, nb_steps, size)

    # ------------------------------ RNN ------------------------------

    rnn_output = mlayers.rnn.LstmRNN(
        size_list=model_param['lstm'],
        return_sequence=True
    )(samples)  # (batch, nb_steps, size)
    rnn_output_steps = mlayers.shapes.Unstack(axis=0)(rnn_output)  # List(nb_steps)[(batch, size)]

    # ------------------------------ Decoding ------------------------------

    decoders = [mlayers.coder2D.Decoder2D(
        decoder_param=model_param_dec,
        shape_before_conv=shape_before_conv_dec,
        dropout=dropout,
        time_stride=time_stride,
        shapes_after_upsize=shapes_before_pooling
    ) for inst in range(nb_instruments)]

    decoded_steps_inst = mlayers.wrapper.l.ApplySameOnList(
        layer=mlayers.wrapper.l.ApplyDifferentLayers(layers=decoders),
        name='decoders'
    )(rnn_output_steps)     # List(nb_steps, nb_instruments)[(batch, step_length, size, channels)]

    last_mono = [mlayers.last.LastInstMono(softmax_axis=-2) for inst in range(nb_instruments)]
    outputs_steps_inst = mlayers.wrapper.l.ApplySameOnList(
        layer=mlayers.wrapper.l.ApplyDifferentOnList(layers=last_mono),
        name='last_mono'
    )(decoded_steps_inst)       # List(nb_steps, nb_instruments)[(batch, step_length, size, channels)]
    outputs_inst_steps = mlayers.shapes.transpose_list(
        outputs_steps_inst,
        axes=(1, 0))        # List(nb_instruments, nb_steps)[batch, step_length, size, channels)]

    outputs = [mlayers.shapes.Stack(axis=0)(o) for o in
               outputs_inst_steps]  # List(nb_instruments)[(batch, nb_steps, step_length, size, channel=1)]
    outputs = [layers.Layer(name=f'Output_{inst}')(outputs[inst]) for inst in range(nb_instruments)]

    model = KerasModel(inputs=inputs_midi + [input_mask], outputs=outputs)

    # ------------------ Losses -----------------
    # Define losses dict for outputs
    losses = {}
    for inst in range(nb_instruments):
        losses[f'Output_{inst}'] = Loss.from_names[loss_options['loss_name']](
            **loss_options
        )

    # Define kld
    if model_options['kld']:
        model.add_loss(kld)

    # ------------------ Metrics -----------------

    # -------------------- Callbacks --------------------
    callbacks = []
    if model_options['kld']:
        callbacks.append(Callbacks.Annealing(
            weight=kld_weight,
            epoch_start=model_options['kld_annealing_start'],
            epoch_stop=model_options['kld_annealing_stop']
        ))

    # ------------------------------ Compile ------------------------------

    model.compile(loss=losses,
                  optimizer=optimizer,
                  metrics=[Loss.metrics.acc_mono()])
    if model_options['kld']:
        model.add_metric(kld, name='kld', aggregation='mean')

    return dict(
        model=model,
        callbacks=callbacks
    )
