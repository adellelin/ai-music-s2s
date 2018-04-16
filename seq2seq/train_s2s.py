import pickle
import atexit
from os.path import dirname, join
import numpy as np
import tensorflow as tf
from midigen.encode import MelodyEncoder

HIDDEN_CODE_SIZE = 20
LEARNING_RATE = 1e-3
L2LAMBDA = 1e-4
L1LAMBDA = 0
CLIP = 5
USE_DROPOUT = False
RECURSIVE_DECODER = False
float_type = tf.float64

training_set_dir = '2_Bar_blues_sets_bass'
training_dir = 'seq2seq_1211_bass_norm'

# training_set_dir = '2_Bar_blues_sets_guitar'
# training_dir = 'seq2seq_1208'
#576103 good, increase l2lambda to 5e-6

with open(join(dirname(__file__), training_set_dir, 'encoder.json'), mode='r') as f:
    mel = MelodyEncoder.from_json(f.read())
with open(join(dirname(__file__), training_set_dir, 'training_set.p'), mode='rb') as f:
    training_set = pickle.load(f)

np.random.seed(1)
with tf.Session() as sess:
    atexit.register(sess.close)

    step = tf.train.get_or_create_global_step()
    call_ohcs = tf.constant(training_set['call_ohcs'], dtype=float_type)
    response_symbols = tf.constant(training_set['response_symbols'], dtype=np.int32)
    response_ohcs = tf.constant(training_set['response_ohcs'], dtype=float_type)
    batch_size, seq_len = response_symbols.shape

    with tf.variable_scope('encoder'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_CODE_SIZE)
        if USE_DROPOUT:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            cell, call_ohcs, time_major=False, dtype=float_type)
        enc_w, enc_b = cell.weights

    with tf.variable_scope('decoder'):
        de_embed_init = np.random.normal(
            loc=0, scale=3e-1,
            size=(HIDDEN_CODE_SIZE, mel.num_symbols + 1))
        de_embed_w = tf.Variable(de_embed_init, name='de_embed_w', trainable=True, dtype=float_type)
        de_embed_b = tf.Variable(np.zeros(mel.num_symbols + 1), dtype=float_type,
                                 name='de_embed_b', trainable=True)

        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_CODE_SIZE)
        if USE_DROPOUT:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)

        go_np = np.zeros((batch_size, mel.num_symbols + 1))
        go_np[:, -1] = 1
        go = tf.constant(go_np, dtype=float_type)

        h_first, c_first = cell(go, enc_state)
        de_first = tf.add(tf.matmul(h_first, de_embed_w), de_embed_b, name='out0')
        dec_outputs = [de_first]
        dec_states = [c_first]
        for i in range(mel.num_time_steps-1):
            if RECURSIVE_DECODER:
                prev_prob = tf.nn.softmax(dec_outputs[-1], dim=1)
                h_cur, state_cur = cell(prev_prob, dec_states[-1])
            else:
                h_cur, state_cur = cell(response_ohcs[:, i, :], dec_states[-1])

            de_cur = tf.add(tf.matmul(h_cur, de_embed_w), de_embed_b, name='out'+str(i+1))
            dec_outputs.append(de_cur)
            dec_states.append(state_cur)
        dec_w, dec_b = cell.weights

    with tf.variable_scope('loss'):
        total_cross_entropy = tf.constant(0.0, dtype=float_type)
        for output_n, dec_output in enumerate(dec_outputs):
            cur_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=response_symbols[:, output_n], logits=dec_output)
            batch_loss = tf.reduce_sum(cur_loss)
            total_cross_entropy += batch_loss

        l2_loss = tf.nn.l2_loss(dec_w) + tf.nn.l2_loss(dec_b) + tf.nn.l2_loss(enc_w) + tf.nn.l2_loss(enc_b)
        l2lambda = tf.placeholder(dtype=float_type)
        l1_loss = tf.norm(dec_w, 1) + tf.norm(dec_b, 1) + tf.norm(enc_w, 1) + tf.norm(enc_b, 1)
        l1lambda = tf.placeholder(dtype=float_type)

        total_loss = total_cross_entropy
        if L2LAMBDA > 0:
            total_loss += l2lambda*l2_loss

        if L1LAMBDA > 0:
            total_loss += l1lambda*l1_loss

    lr = tf.placeholder(dtype=float_type)
    optimizer = tf.train.AdamOptimizer(lr)
    gradients, variables = zip(*optimizer.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, CLIP)
    optimize = optimizer.apply_gradients(zip(gradients, variables), global_step=step)

    tf.summary.scalar('cross_entropy', total_cross_entropy)
    tf.summary.scalar('l2_loss', l2_loss)
    tf.summary.scalar('l2_lambda', l2lambda)
    tf.summary.scalar('l1_loss', l1_loss)
    tf.summary.scalar('l1_lambda', l1lambda)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('total_loss', total_loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(join(dirname(__file__), training_dir), graph=sess.graph)

    saver = tf.train.Saver(tf.trainable_variables().append(step))

    cp = tf.train.latest_checkpoint(join(dirname(__file__), training_dir))
    if cp is None:
        print('initializing')
        tf.global_variables_initializer().run()
    else:
        print('restoring: ' + cp)
        saver.restore(sess, cp)

    summary, cur_step = sess.run([merged, step], feed_dict={lr: 0.0, l2lambda: 0.0, l1lambda: 0.0})
    if cur_step == 0:
        writer.add_summary(summary, global_step=cur_step)
        saver.save(sess, join(dirname(__file__), training_dir, 'test_model'), global_step=step, write_meta_graph=True)

    i = 0

    def norm(x):
        # p = np.percentile(np.abs(x), 95)
        l = 1e-9
        # mask =
        return (x-l*np.sign(x))*(x > l)

    while True:
        feed_dict = {lr: LEARNING_RATE, l2lambda: L2LAMBDA, l1lambda: L1LAMBDA}
        summary, cur_step, _ = sess.run([merged, step, optimize], feed_dict=feed_dict)
        if i % 1 == 0:
            enc_w_cur, dec_w_cur, de_embed_w_cur = sess.run([enc_w, dec_w, de_embed_w])
            enc_w.load(norm(enc_w_cur), session=sess)
            dec_w.load(norm(dec_w_cur), session=sess)
            # de_embed_w.load(norm(de_embed_w_cur), session=sess)

        if i % 1000 == 0:
            writer.add_summary(summary, global_step=cur_step)
            saver.save(sess, join(dirname(__file__), training_dir, 'test_model'),
                       global_step=step, write_meta_graph=False)

        i += 1
