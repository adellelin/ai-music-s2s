import logging
import os
import tensorflow as tf
import magenta
from magenta.models.shared.events_rnn_graph import build_graph
from magenta.models.melody_rnn.melody_rnn_sequence_generator import MelodyRnnSequenceGenerator
from magenta.models.melody_rnn.melody_rnn_model import MelodyRnnModel

logging.basicConfig(level=logging.DEBUG)

BUNDLE_NAME = 'attention_rnn'

config = magenta.models.melody_rnn.melody_rnn_model.default_configs[BUNDLE_NAME]
bundle_file = magenta.music.read_bundle_file(os.path.abspath(BUNDLE_NAME+'.mag'))
steps_per_quarter = 4

generator = MelodyRnnSequenceGenerator(
    model=MelodyRnnModel(config),
    details=config.details,
    steps_per_quarter=steps_per_quarter,
    bundle=bundle_file)

generator.create_bundle_file('/Users/mdpicket/Documents/tmp/bundle')
exit()

with tf.Session() as sess:
    graph = build_graph('generate', config)
    writer = tf.summary.FileWriter(logdir='/Users/mdpicket/Documents/tmp',
                                   graph=graph)
    writer.flush()

    with tf.Graph() as graph:
     saver = tf.train.Saver(tf.global_variables())
     saver.save(sess, '/Users/mdpicket/Documents/tmp/model.saver')
