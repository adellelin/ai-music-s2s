import pickle
from os.path import abspath, join, dirname
import numpy as np
import pretty_midi as pm
import tensorflow as tf
from midigen.encode import MelodyEncoder, concat

training_set_dir = '2_Bar_blues_sets_bass'
training_dir = 'seq2seq_1211_bass_norm'

# training_set_dir = '2_Bar_blues_sets_guitar'
# training_dir = 'seq2seq_1208'

with open(join(dirname(__file__), training_set_dir, 'encoder.json'), mode='r') as f:
    mel = MelodyEncoder.from_json(f.read())
with open(join(dirname(__file__), training_set_dir, 'validation_set.p'), mode='rb') as f:
# with open(join(dirname(__file__), training_set_dir, 'training_set.p'), mode='rb') as f:
    validation_set = pickle.load(f)

builder_path = join(dirname(__file__), training_dir, 'eval/model')
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [], builder_path)
    for v in tf.global_variables():
        print(v.name)
        print(sess.run(v))

    # TODO: use builder signatures here?
    outputs = []
    for i in range(mel.num_time_steps):
        outputs.append(sess.graph.get_tensor_by_name('decoder/out'+str(i)+':0'))
    call_ohcs = sess.graph.get_tensor_by_name('call_ohcs:0')

    batch_size = validation_set['call_ohcs'].shape[0]

    output_symbols = np.empty((batch_size, mel.num_time_steps), dtype=np.int32)
    for set_n in range(batch_size):
        cur_call = validation_set['call_ohcs'][set_n].reshape((1, mel.num_time_steps, mel.num_symbols))
        cur_outputs = sess.run(outputs, {call_ohcs: cur_call})
        for seq_n, output in enumerate(cur_outputs):
            output_symbols[set_n, seq_n] = np.argmax(output)

call_prog = pm.instrument_name_to_program('Acoustic Grand Piano')
gen_prog = pm.instrument_name_to_program('Electric Piano 1')

midis = []
for batch_n in range(batch_size):
    call_midi = mel.decode(
        validation_set['call_symbols'][batch_n],
        program=call_prog)

    gen_midi = mel.decode(
        output_symbols[batch_n],
        program=gen_prog)

    midis.append(call_midi)
    midis.append(gen_midi)

midi_full = concat(midis, min_len=mel.total_time)
midi_full.write(abspath('out/full.mid'))
