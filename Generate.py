import pickle
import numpy as np

from music21 import instrument
from music21 import note
from music21 import chord
from music21 import stream

from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))

note_to_int = dict(
    (note, number)
    for number, note in enumerate(pitchnames)
)

int_to_note = dict(
    (number, note)
    for number, note in enumerate(pitchnames)
)

sequence_length = 5
network_input = []

for i in range(0, len(notes)-sequence_length):
    sequence = notes[i:i+sequence_length]
    network_input.append(
        [note_to_int[n] for n in sequence]
    )

print("Patterns:", len(network_input))

if len(network_input) == 0:
    print("No patterns found. Increase MIDI data or reduce sequence_length.")
    exit()

start = np.random.randint(0, len(network_input))

pattern = network_input[start]



start = np.random.randint(0, len(network_input))

pattern = network_input[start]

prediction_output = []

# Generate 500 notes
for note_index in range(500):

    prediction_input = np.reshape(
        pattern,
        (1, len(pattern), 1)
    )

    prediction_input = prediction_input / float(len(pitchnames))

    prediction = model.predict(
        prediction_input,
        verbose=0
    )

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern.append(index)

    pattern = pattern[1:]

# Convert to MIDI
offset = 0
output_notes = []

for pattern in prediction_output:

    if '.' in pattern or pattern.isdigit():

        notes_in_chord = pattern.split('.')
        chord_notes = []

        for current_note in notes_in_chord:

            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()

            chord_notes.append(new_note)

        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset

        output_notes.append(new_chord)

    else:

        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()

        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write(
    'midi',
    fp='generated_music.mid'
)

print("Music Generated Successfully")