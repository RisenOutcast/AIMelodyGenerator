import glob

from Generation import MusicGeneration
from Generation import MusicManager

filenames = glob.glob(str('./midi/**/**/*.mid*'))
print('Number of files:', len(filenames))

sample_file = filenames[1]

# Get instrument details: [instrument, instrument's name]
instrument = MusicManager.GetInstruments(sample_file)
MusicManager.ExtractNotes(instrument[0])

# Generate a example midi file
raw_notes = MusicManager.CreateExampleMidiFile(sample_file, instrument)

train_ds = MusicGeneration.create_training_dataset()

model = MusicManager.CreateModel(train_ds)

MusicManager.TrainModel(train_ds, model)

generated_notes = MusicManager.GenerateNotes(raw_notes, model)

out_file = 'output.mid'
out_pm = MusicGeneration.notes_to_midi(
    generated_notes, out_file=out_file, instrument_name=instrument[1])
