import glob
import pickle
import numpy as np

from music21 import converter, instrument, note, chord

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
notes = []

notes = []

# Read MIDI files
for file in glob.glob("dataset/*.mid"):

    print("Reading:", file)

    midi = converter.parse(file)

    try:
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

notes = []

# Read MIDI files
for file in glob.glob("dataset/*.mid"):

    midi = converter.parse(file)

    try:
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

# Save extracted notes
with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

# Create sequences
sequence_length = 5
if len(notes) <= sequence_length:
    print(f"Not enough notes. Found {len(notes)} notes, need more than {sequence_length}.")
    exit()
    

pitchnames = sorted(set(notes))

note_to_int = dict(
    (note, number)
    for number, note in enumerate(pitchnames)
)

network_input = []
network_output = []

for i in range(0, len(notes)-sequence_length):

    sequence_in = notes[i:i+sequence_length]
    sequence_out = notes[i+sequence_length]

    network_input.append(
        [note_to_int[char] for char in sequence_in]
    )

    network_output.append(
        note_to_int[sequence_out]
    )

n_patterns = len(network_input)

network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

network_input = network_input / float(len(pitchnames))

network_output = to_categorical(network_output)

# Build Model
model = Sequential()

model.add(
    LSTM(
        512,
        input_shape=(network_input.shape[1],
                     network_input.shape[2]),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(
        512,
        return_sequences=False
    )
)

model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))

model.add(Dense(
    network_output.shape[1],
    activation='softmax'
))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)

# Train Model
model.fit(
    network_input,
    network_output,
    epochs=50,
    batch_size=64
)

model.save("model.h5")

print("Training Complete")