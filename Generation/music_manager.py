import glob
import pretty_midi
import numpy as np
import pandas as pd
import tensorflow as tf

from Generation import MusicGeneration
from config import seed, seq_length, vocab_size, key_order, temperature, num_predictions, learning_rate, epochs

filenames = glob.glob(str('./midi/**/**/*.mid*'))

class MusicManager:
    def GetInstruments(sample_file):
        print(sample_file)

        pm = pretty_midi.PrettyMIDI(sample_file)

        print('Number of instruments:', len(pm.instruments))
        instrument = pm.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        print('Instrument name:', instrument_name)
        return [instrument, instrument_name]
    
    def ExtractNotes(instrument):
        for i, note in enumerate(instrument.notes[:10]):
            note_name = pretty_midi.note_number_to_name(note.pitch)
            duration = note.end - note.start
            print(f'{i}: pitch={note.pitch}, note_name={note_name},'
                    f' duration={duration:.4f}')
            
    def CreateExampleMidiFile(sample_file, instrument):
        raw_notes = MusicGeneration.midi_to_notes(sample_file)
        raw_notes.head()

        example_file = 'example.mid'
        example_pm = MusicGeneration.notes_to_midi(raw_notes, out_file=example_file, instrument_name=instrument[1])
        return raw_notes
    
    def CreateModel(train_ds):
        input_shape = (seq_length, 3)

        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(128)(inputs)

        outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
        }

        model = tf.keras.Model(inputs, outputs)

        loss = {
            'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            'step': MusicGeneration.mse_with_positive_pressure,
            'duration': MusicGeneration.mse_with_positive_pressure,
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(loss=loss, optimizer=optimizer)

        model.summary()

        losses = model.evaluate(train_ds, return_dict=True)
        losses

        model.compile(
            loss=loss,
            loss_weights={
                'pitch': 0.05,
                'step': 1.0,
                'duration':1.0,
            },
            optimizer=optimizer,
        )

        model.evaluate(train_ds, return_dict=True)
        return model
    
    def TrainModel(train_ds, model):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./training_checkpoints/ckpt_{epoch}',
                save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True),
            ]

        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

    def GenerateNotes(raw_notes, model):
        sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training
        # sequences
        input_notes = (
            sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

        generated_notes = []
        prev_start = 0
        for _ in range(num_predictions):
            pitch, step, duration = MusicGeneration.predict_next_note(input_notes, model, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(
            generated_notes, columns=(*key_order, 'start', 'end'))
        
        return generated_notes