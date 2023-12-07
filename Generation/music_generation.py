import collections
import glob
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from config import seed, seq_length, vocab_size, key_order

tf.random.set_seed(seed)
np.random.seed(seed)
filenames = glob.glob(str('./midi/**/**/*.mid*'))

class MusicGeneration:
    def midi_to_notes(midi_file: str) -> pd.DataFrame:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

    def notes_to_midi(
        notes: pd.DataFrame,
        out_file: str, 
        instrument_name: str,
        velocity: int = 100,  # note loudness
        ) -> pretty_midi.PrettyMIDI:

        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program(
                instrument_name))

        prev_start = 0
        for i, note in notes.iterrows():
            start = float(prev_start + note['step'])
            end = float(start + note['duration'])
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(note['pitch']),
                start=start,
                end=end,
            )
            instrument.notes.append(note)
            prev_start = start

        pm.instruments.append(instrument)
        pm.write(out_file)
        return pm

    def create_training_dataset():
        num_files = 5
        all_notes = []
        for f in filenames[:num_files]:
            notes = MusicGeneration.midi_to_notes(f)
            all_notes.append(notes)

        all_notes = pd.concat(all_notes)

        n_notes = len(all_notes)
        print('Number of notes parsed:', n_notes)

        train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

        notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
        notes_ds.element_spec

        seq_ds = MusicGeneration.create_sequences(notes_ds, seq_length, vocab_size)
        seq_ds.element_spec

        for seq, target in seq_ds.take(1):
            print('sequence shape:', seq.shape)
            print('sequence elements (first 10):', seq[0: 10])
            print()
            print('target:', target)

        batch_size = 64
        buffer_size = n_notes - seq_length  # the number of items in the dataset
        train_ds = (seq_ds
                    .shuffle(buffer_size)
                    .batch(batch_size, drop_remainder=True)
                    .cache()
                    .prefetch(tf.data.experimental.AUTOTUNE))

        return train_ds

    def create_sequences(
            dataset: tf.data.Dataset, 
            seq_length: int,
            vocab_size = 128,
        ) -> tf.data.Dataset:
        """Returns TF Dataset of sequence and label examples."""
        seq_length = seq_length+1

        # Take 1 extra for the labels
        windows = dataset.window(seq_length, shift=1, stride=1,
                                    drop_remainder=True)

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(seq_length, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        # Normalize note pitch
        def scale_pitch(x):
            x = x/[vocab_size,1.0,1.0]
            return x

        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            labels_dense = sequences[-1]
            labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

            return scale_pitch(inputs), labels

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
        mse = (y_true - y_pred) ** 2
        positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
        return tf.reduce_mean(mse + positive_pressure)

    def predict_next_note(
        notes: np.ndarray, 
        model: tf.keras.Model, 
        temperature: float = 1.0) -> tuple[int, float, float]:
        """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = model.predict(inputs)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)

        # `step` and `duration` values should be non-negative
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration)