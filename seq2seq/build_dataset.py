# Import the os module, for the os.walk function
import os
import pickle
from os.path import abspath, join
import logging
import numpy as np
import pretty_midi as pm
from midigen.encode import MelodyEncoder, concat

logger = logging.getLogger('build_dataset')
logging.basicConfig()

# data_dir = abspath('2_bar_blues_sets_guitar')
# encoder = MelodyEncoder(
#     [55, 58, 60, 63, 65, 66, 67, 70, 72, 75, 77, 78, 79, 82, 84],
#     8, 2, 2)

data_dir = abspath('2_bar_blues_sets_bass')
encoder = MelodyEncoder(
#     [12, 15, 17, 18, 19, 22, 24, 27],  # supplied note list
    [36, 39, 43, 41, 42, 46, 48],        # actual notes in the set
    8, 2, 2)

np.random.seed(1)
validation_ratio = 0.05

calls = []
responses = []
cr_midis = []
for dirName, subdirList, fileList in os.walk(data_dir):
    prev_encoding = None
    cur_cr = {}
    for fname in fileList:
        if fname.endswith('.mid') and fname != 'full.mid':
            num = int(fname.replace('.mid', '').replace(' ', '_').split('_')[-1])
            full_path = join(dirName, fname)
            try:
                midi = pm.PrettyMIDI(full_path)
                assert len(midi.instruments[0].notes) > 3
                cur_cr[num] = encoder.encode(midi)
            except (AssertionError, IOError) as e:
                logger.exception(join(dirName, fname))

    for k in cur_cr.keys():
        if k % 2 == 1:
            try:
                cur_call = cur_cr[k]
                cur_response = cur_cr[k + 1]
                responses.append(cur_response)
                calls.append(cur_call)
                cr_midis.append(encoder.decode(cur_call, pm.instrument_name_to_program('Acoustic Grand Piano')))
                cr_midis.append(encoder.decode(cur_response, pm.instrument_name_to_program('Electric Piano 1')))

            except KeyError:
                pass

midi_full = concat(cr_midis, min_len=encoder.total_time)
midi_full.write(join(data_dir, 'full.mid'))

all_symbols = []
all_symbols.extend(calls)
all_symbols.extend(responses)
all_symbols = np.concatenate(all_symbols)

all_pitch_symbols = all_symbols[np.where(np.logical_and(all_symbols != 0, all_symbols != 1))]
uniq_symbols, pitch_count = np.unique(all_pitch_symbols, return_counts=True)
pitch = [encoder.decoder_lut[sym] for sym in uniq_symbols]
pitch_names = [pm.note_number_to_name(p) for p in pitch]

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.figure(1, figsize=(10, 5))
plt.bar(np.arange(len(pitch_names)), pitch_count, align='center', alpha=0.5)
plt.ylabel('Note count in data set')
plt.xlabel('Notes')
plt.xticks(np.arange(len(pitch_names)), pitch_names)
plt.savefig(join(data_dir, 'hist.png'))
plt.close()

assert len(calls) == len(responses)
num_pairs = len(calls)
validation_num = int(num_pairs*validation_ratio)
training_num = num_pairs-validation_num

call_symbols = np.array(calls, dtype=np.uint8)
response_symbols = np.array(responses, dtype=np.uint8)

rand_ordering = np.argsort(np.random.random(num_pairs))
training_indices = rand_ordering[:training_num]
validation_indices = rand_ordering[training_num:]

training_call_symbols = call_symbols[training_indices]
training_response_symbols = response_symbols[training_indices]
validation_call_symbols = call_symbols[validation_indices]
validation_response_symbols = response_symbols[validation_indices]


def to_ohc(arg, extra_symbols):
    batch_size, seq_len = arg.shape
    ohcs = np.zeros((batch_size, seq_len, encoder.num_symbols+extra_symbols), dtype=np.float32)
    for batch_n in range(batch_size):
        for step_n in range(seq_len):
            ohcs[batch_n, step_n, arg[batch_n, step_n]] = 1.0
    return ohcs


training_set = {
    'call_symbols': training_call_symbols,
    'call_ohcs': to_ohc(training_call_symbols, 0),
    'response_symbols': training_response_symbols,
    'response_ohcs': to_ohc(training_response_symbols, 1)
}

validation_set = {
    'call_symbols': validation_call_symbols,
    'call_ohcs': to_ohc(validation_call_symbols, 0),
    'response_symbols': validation_response_symbols,
    # 'response_ohcs': to_ohc(validation_response_symbols, 0)
}


with open(join(data_dir, 'training_set.p'), mode='wb') as f:
    pickle.dump(training_set, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(join(data_dir, 'validation_set.p'), mode='wb') as f:
    pickle.dump(validation_set, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(join(data_dir, 'encoder.json'), mode='w') as f:
    f.write(encoder.to_json())
