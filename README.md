# Drum Pattern Generation with Transformer Inpainting

Submitted for ECS7022P Computational Creativity at Queen Mary University of London

## Structure

The `main.ipynb` file contains the majority of this projects implementation, including an interface developed as a playground for exprimenting with the system, as well as the code used to train it. Three other files contain some class and function definitions, split to clean up the main notebook. These are packaged as a python project (see the `pyproject.toml` file) and include:
- `tokeniser.py` - Implementations of a number of tokenisation methods which were experimented with (only `DrumSequenceEncoder` is used in the final system)
- `modules.py` - Implementation of the custom transformer layers which include a relative position encoding and associated attention blocks.
- `utils.py` - Helper methods for processing, displaying and playing MIDI data.

## Examples

Below are some example outputs of the system, paired with the input patterns which were used as prompts.

### Backbeat

Original: [backbeat_original.wav](output/audio/backbeat_original.wav)

Subtle Variation: [backbeat_minimal.wav](output/audio/backbeat_minimal.wav)

Pocket Beat: [backbeat_tight.wav](output/audio/backbeat_tight.wav)

Lots of shifts: [backbeat_shifts.wav](output/audio/backbeat_shifts.wav)

Dense sampling: [backbeat_dense.wav](output/audio/backbeat_dense.wav)

### Amen

Original: [amen_original](output/audio/amen_original.wav)

Shifting: [amen_shifts](output/audio/amen_shifts.wav)

Breakbeats: [amen_dense](output/audio/amen_dense.wav)

<audio controls src="output/audio/amen_original.wav" title="Original"></audio>
<audio controls src="output/audio/amen_shifts.wav" title="Shifts"></audio>
<audio controls src="output/audio/amen_dense.wav" title="Dense"></audio>


### Be my Baby

Original: [be_my_baby_original](output/audio/be_my_baby_original.wav)

Syncopated: [be_my_baby_syncopated](output/audio/be_my_baby_syncopated.wav)

Improvised: [be_my_baby_improvised](output/audio/be_my_baby_improvised.wav)

<audio controls src="output/audio/be_my_baby_original.wav" title="Original"></audio>
<audio controls src="output/audio/be_my_baby_syncopated.wav" title="Syncopated"></audio>
<audio controls src="output/audio/be_my_baby_improvised.wav" title="Improvised"></audio>