# AI Melody Generation :notes:
Program for generating melodies using AI

### Prerequisite
Have Python (version >3.10) installed.

Create a folder called `midi` in the main directory.

Extract your training data to the `midi`-folder, I used the MAESTRO dataset from TensorFlow: [maestro-v3.0.0-midi.zip](https://magenta.tensorflow.org/datasets/maestro#download)


### How to run?

Clone the repo and run these commands

Install dependencies
`python setup.py install`

Run the program
`python main.py`

### Output:

The program generates a new melody into `output.mid`, which you can use as it is or import to music production software for further usage.

### Configuring

All training configuration variables can be found in `config.py`
