import atexit
from os.path import join, dirname
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from midigen.encode import MelodyEncoder

HIDDEN_CODE_SIZE = 20
train_type = np.float64
float_type = np.float16

training_set_dir = '2_Bar_blues_sets_bass'
training_dir = 'seq2seq_1211_bass_norm'

# training_set_dir = '2_Bar_blues_sets_guitar'
# training_dir = 'seq2seq_1208'


with open(join(dirname(__file__), training_set_dir, 'encoder.json'), mode='r') as f:
    mel = MelodyEncoder.from_json(f.read())

get = {
    'decoder/de_embed_w': (HIDDEN_CODE_SIZE, mel.num_symbols+1),
    'decoder/de_embed_b': mel.num_symbols + 1,
    'encoder/rnn/basic_lstm_cell/kernel': (mel.num_symbols + HIDDEN_CODE_SIZE, 4*HIDDEN_CODE_SIZE),
    'encoder/rnn/basic_lstm_cell/bias': 4*HIDDEN_CODE_SIZE,
    'decoder/basic_lstm_cell/kernel': (mel.num_symbols+1+HIDDEN_CODE_SIZE, 4*HIDDEN_CODE_SIZE),
    'decoder/basic_lstm_cell/bias': 4*HIDDEN_CODE_SIZE
}
# load all checkpointed variables to convert them to float_type
with tf.Session() as sess:
    variables = {}
    for name, shape in get.iteritems():
        variables[name] = tf.Variable(
            np.zeros(shape, dtype=train_type), name=name)

    saver = tf.train.Saver(variables.values())
    cp = tf.train.latest_checkpoint(join(dirname(__file__), training_dir))
    saver.restore(sess, cp)

    arrays = {}
    for name, var in variables.iteritems():
        arrays[name] = var.eval().astype(float_type)
sess.close()
tf.reset_default_graph()

with tf.Session() as sess:
    atexit.register(sess.close)

    batch_size = 1
    call_ohcs = tf.placeholder(
        dtype=float_type,
        shape=(batch_size, mel.num_time_steps, mel.num_symbols),
        name='call_ohcs')

    with tf.variable_scope('encoder/rnn') as vs:

        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_CODE_SIZE)
        enc_state = cell.zero_state(batch_size, float_type)
        for i in range(mel.num_time_steps):
            enc_outputs, enc_state = cell(call_ohcs[:, i, :], enc_state)
        enc_w, enc_b = cell.weights

    with tf.variable_scope('decoder'):
        de_embed_w = tf.Variable(arrays['decoder/de_embed_w'], name='de_embed_w')
        de_embed_b = tf.Variable(arrays['decoder/de_embed_b'], name='de_embed_b')

        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_CODE_SIZE)
        go_np = np.zeros((batch_size, mel.num_symbols + 1), dtype=float_type)
        go_np[:, -1] = 1
        go = tf.constant(go_np)

        h_first, c_first = cell(go, enc_state)
        dec_w, dec_b = cell.weights

        de_first = tf.add(tf.matmul(h_first, de_embed_w), de_embed_b, name='out0')
        dec_outputs = [de_first]
        dec_states = [c_first]
        for i in range(mel.num_time_steps-1):
            prev_prob = tf.nn.softmax(dec_outputs[-1], dim=1)
            h_cur, state_cur = cell(prev_prob, dec_states[-1])

            de_cur = tf.add(tf.matmul(h_cur, de_embed_w), de_embed_b, name='out'+str(i+1))
            dec_outputs.append(de_cur)
            dec_states.append(state_cur)

    sess.run(tf.global_variables_initializer())
    enc_w.load(arrays['encoder/rnn/basic_lstm_cell/kernel'], session=sess)
    enc_b.load(arrays['encoder/rnn/basic_lstm_cell/bias'], session=sess)
    dec_w.load(arrays['decoder/basic_lstm_cell/kernel'], session=sess)
    dec_b.load(arrays['decoder/basic_lstm_cell/bias'], session=sess)

    shutil.rmtree(join(dirname(__file__), training_dir, 'eval'))

    builder = saved_model_builder.SavedModelBuilder(join(dirname(__file__), training_dir, 'eval/model'))
    builder.add_meta_graph_and_variables(
        sess, [])
    builder.save()

    with open(join(dirname(__file__), training_dir, 'eval/model', 'encoder.json'), mode='w') as f:
        f.write(mel.to_json())
