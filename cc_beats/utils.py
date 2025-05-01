import torchaudio

import symusic
import symusic.types

from midi2audio import FluidSynth
import IPython.display as ipd

import tempfile
import os

def midi_to_audio_display(fs: FluidSynth, midi: symusic.types.Score | str):
    file = midi
    if isinstance(file, symusic.types.Score):
        for track in midi.tracks:
            if not track.is_drum:
                track.is_drum = True
        # Create a temporary file with .mid extension
        midi_temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
        midi_temp_filename = midi_temp_file.name
        midi_temp_file.close()
        
        # Save the MIDI object to the temp file
        midi.dump_midi(midi_temp_filename)
        file = midi_temp_filename

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as audio_temp_file:
        # Convert MIDI to audio and display
        fs.midi_to_audio(file, audio_temp_file.name)
        display_audio = ipd.Audio(audio_temp_file.name)
        
        # Clean up the temp file if we created one
        if isinstance(midi, symusic.types.Score) and 'temp_filename' in locals():
            os.remove(midi_temp_filename)
            
    return display_audio

def midi_to_audio_tensor(fs: FluidSynth, midi: symusic.types.Score | str):
    file = midi
    if isinstance(file, symusic.types.Score):
        for track in midi.tracks:
            if not track.is_drum:
                track.is_drum = True
        # Create a temporary file with .mid extension
        midi_temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
        midi_temp_filename = midi_temp_file.name
        midi_temp_file.close()
        
        # Save the MIDI object to the temp file
        midi.dump_midi(midi_temp_filename)
        file = midi_temp_filename
    

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as audio_temp_file:
        # Convert MIDI to audio and display
        fs.midi_to_audio(file, audio_temp_file.name)
        audio, sr = torchaudio.load(audio_temp_file.name)
        
        # Clean up the temp file if we created one
        if isinstance(midi, symusic.types.Score) and 'temp_filename' in locals():
            os.remove(midi_temp_filename)
    
    
    audio = audio.mean(dim=0)
    return audio, sr