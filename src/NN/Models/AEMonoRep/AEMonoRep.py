import tensorflow as tf

from src import GlobalVariables as g
import src.NN.layers as mlayers
from src.NN.Models.KerasModel import KerasModel
import src.NN.shapes.convolution as s_conv
import src.NN.shapes.time as s_time
from src.eval_string import eval_object
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
        dropout=g.nn.dropout,
    )
    dictionaries.set_default(model_options, model_options_default)

    loss_options_default = dict(
        loss_name='mono',
        l_scale=g.loss.l_scale,
        l_rhythm=g.loss.l_rhythm,
        l_scale_cost=g.loss.l_scale_cost,
        l_rhythm_cost=g.loss.l_rhythm_cost,
        take_all_step_rhythm=g.loss.take_all_step_rhythm
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

    inputs_inst_step = [mlayers.shapes.Unstack(axis=0)(input_inst) for input_inst in
                        inputs_midi]  # List(nb_instruments, nb_steps)[(batch, step_length, size, channels)]

    # ------------------------------ Encoding ------------------------------

    encoders = [mlayers.coder2D.Encoder2D(
        encoder_param=model_param,
        dropout=dropout,
        time_stride=time_stride,
        time_distributed=False
    ) for inst in range(nb_instruments)]

    encoded_step_inst = [[encoders[inst](inputs_inst_step[inst][step]) for inst in range(nb_instruments)] for step in
                         range(nb_steps)]  # List(nb_inst, nb_instruments)[(batch, size)]

    encoded_steps = [layers.concatenate(encoded_step_inst[step]) for step in
                     range(nb_steps)]  # List(nb_steps)[(batch, size)]
    encoded = mlayers.shapes.Stack(axis=0)(encoded_steps)  # (batch, nb_steps, size)

    # ------------------------------ RNN ------------------------------

    rnn_output = mlayers.rnn.LstmRNN(
        size_list=model_param['lstm'],
        return_sequence=True
    )(encoded)  # (batch, nb_steps, size)
    rnn_output_steps = mlayers.shapes.Unstack(axis=0)(rnn_output)  # List(nb_steps)[(batch, size)]

    # ------------------------------ Decoding ------------------------------

    decoders = [mlayers.coder2D.Decoder2D(
        decoder_param=model_param_dec,
        shape_before_conv=shape_before_conv_dec,
        dropout=dropout,
        time_stride=time_stride,
        shapes_after_upsize=shapes_before_pooling
    ) for inst in range(nb_instruments)]

    decoded_inst_steps = [
        [
            decoders[inst](rnn_output_steps[step])
            for step in range(nb_steps)
        ]
        for inst in range(nb_instruments)
    ]  # List(nb_instruments)[List(nb_steps)[batch, step_length, size, channels=1)]]
    last_mono = [mlayers.last.LastInstMono(softmax_axis=-2) for inst in range(nb_instruments)]
    outputs_inst_steps = [
        [
            last_mono[inst](decoded_inst_steps[inst][step])
            for step in range(nb_steps)
        ]
        for inst in range(nb_instruments)
    ]  # List(nb_instruments)[List(nb_steps)[(batch, step_length, size, channels=1)]]
    outputs = [mlayers.shapes.Stack(axis=0)(o) for o in
               outputs_inst_steps]  # List(nb_instruments)[(batch, nb_steps, step_length, size, channel=1)]
    outputs = [layers.Layer(name=f'Output_{inst}')(outputs[inst]) for inst in range(nb_instruments)]

    model = KerasModel(inputs=inputs_midi, outputs=outputs)

    # ------------------ Losses -----------------
    # Define losses dict for outputs
    losses = {}
    for inst in range(nb_instruments):
        losses[f'Output_{inst}'] = Loss.from_names[loss_options['loss_name']](
            **loss_options
        )
    # ------------------ Metrics -----------------

    # ------------------------------ Compile ------------------------------

    model.compile(loss=losses,
                  optimizer=optimizer,
                  metrics=[Loss.metrics.acc_mono()])
    model.build([(None, *midi_shape) for inst in range(nb_instruments)])

    return dict(
        model=model,
    )
