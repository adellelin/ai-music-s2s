# set up environment

```
conda create -n ces-ai-jazz python=2.7 jupyter
source activate ces-ai-jazz
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py2-none-any.whl
pip install --upgrade flask
pip install --upgrade magenta
```

if building the magenta tools directly you must install bazel:
https://docs.bazel.build/versions/master/install-os-x.html


# activate environment
```
source activate ces-ai-jazz
```

# run training
based on tutorial at https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md
https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn

```
convert_dir_to_note_sequences --input_dir='/Users/mdpicket/Documents/ces-ai-jazz/matthew/AI C_Blues_120BPM_All_Melodies' --output_file='~/Documents/ces-ai-jazz/matthew/notesequences.tfrecord'

melody_rnn_create_dataset \
--config='attention_rnn' \
--input='~/Documents/ces-ai-jazz/matthew/notesequences.tfrecord' \
--output_dir='~/Documents/ces-ai-jazz/matthew/melody_rnn/sequence_examples'

melody_rnn_train \
--config=attention_rnn \
--run_dir='~/Documents/ces-ai-jazz/matthew/melody_rnn/run1' \
--sequence_example_file='~/Documents/ces-ai-jazz/matthew/melody_rnn/sequence_examples/training_melodies.tfrecord' \
--num_training_steps=10
```

to generate a bundle file from the trained checkpoint file:
```
melody_rnn_generate \
--config=attention_rnn \
--run_dir='~/Documents/ces-ai-jazz/matthew/melody_rnn/run1' \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--bundle_file='~/Documents/ces-ai-jazz/matthew/melody_rnn/attention_rnn.mag' \
--save_generator_bundle
```

# magenta notes
tensorflow graph defined in magenta.models.shared.events_rnn_graph.py

midi to NoteSequence proto conversion in magenta.music.midi_io.py.
NoteSequence is an intermediate protobuf format defined in magenta.protobuf.music.proto.
This format is similar to midi in concept.

sequence proto encoded to model inputs by the encoder/decoders defined
in magenta.music.encoder_decoder.py

```
export PATH="$PATH:$HOME/bin"
bazel build magenta/models/melody_rnn:melody_rnn_train
./bazel-bin/magenta/models/melody_rnn/melody_rnn_train
```
