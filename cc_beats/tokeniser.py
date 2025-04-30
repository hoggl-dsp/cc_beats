import torch
import symusic
import symusic.types
from typing import Union
import os
import tqdm

ROLAND_DRUM_MAPPING = {
    35  : 36, # Kick 2
    36	: 36, # Kick	
    37	: 38, # Snare X-Stick
    38	: 38, # Snare (Head)
    40	: 38, # Snare (Rim)
    48	: 50, # Tom 1
    50	: 50, # Tom 1 (Rim)
    45  : 47, # Tom 2
    47	: 47, # Tom 2 (Rim)
    43	: 43, # Tom 3 (Head)
    58	: 43, # Tom 3 (Rim)
    46	: 46, # HH Open (Bow)
    26	: 46, # HH Open (Edge)
    42	: 42, # HH Closed (Bow)
    22	: 42, # HH Closed (Edge)
    44	: 42, # HH Pedal
    49	: 49, # Crash 1 (Bow)
    55	: 49, # Crash 1 (Edge)
    57	: 49, # Crash 2 (Bow)
    52	: 49, # Crash 2 (Edge)
    51	: 51, # Ride (Bow)
    59	: 51, # Ride (Edge)
    53	: 51, # Ride (Bell)
}

class DrumSequenceTokeniser:
    def __init__(self, subdivision: int = 16, velocity_bands: int = 4, drum_mapping: dict[int, int] = ROLAND_DRUM_MAPPING):
        self.subdivision = subdivision
        self.velocity_bands = velocity_bands
        self.drum_mapping = drum_mapping
        self.tpq = 480

        self.vocab = {
            '<pad>': 0,
            '<mask>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<rest>': 4,
        }
        self.special_token_count = len(self.vocab)

        for pitch in drum_mapping.values():
            for vel in range(self.velocity_bands):
                key = f'p{pitch}_v{vel}'
                if key not in self.vocab:
                    self.vocab[key] = len(self.vocab)
    
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, key: str):
        return self.vocab[key]

    def encode(self, midi: Union[str, symusic.types.Score]) -> list[int] | None:
        # Load MIDI if needed
        if isinstance(midi, str):
            midi = symusic.Score.from_file(midi)
        
        drum_tracks = [track for track in midi.tracks if track.is_drum]

        # Filter out sparse midi files
        if all([len(track.notes) <= 10 for track in drum_tracks]):
            return
        
        ticks_per_subdiv = midi.ticks_per_quarter * (4. / self.subdivision)

        grid = dict[int, dict[int, int]]()
        last_note = max(track.notes[-1].time for track in drum_tracks if len(track.notes) > 0)
        for index in range(int((last_note + 0.5 * ticks_per_subdiv) / ticks_per_subdiv) + 1):
            grid[index] = {}
        
        for drum_track in drum_tracks:
            for note in drum_track.notes:
                if not note.pitch in self.drum_mapping:
                    continue

                grid_index = int((note.time + 0.5 * ticks_per_subdiv) / ticks_per_subdiv)
                
                note_pitch = self.drum_mapping[note.pitch]
                note_vel = int(note.velocity * ((self.velocity_bands) / 128))

                # We could have multiple of the same drum hits in same subdivision
                # For simplicity, take the largest velocity of these
                if note_pitch in grid[grid_index]:
                    grid[grid_index][note_pitch] = max(grid[grid_index][note_pitch], note_vel)
                else:
                    grid[grid_index][note_pitch] = note_vel

        tok_sequence = []
        tok_sequence.append('<bos>')

        for note_map in grid.values():
            if len(note_map) == 0:
                tok_sequence.append('<rest>')
                continue
            
            tok_combo = []
            for pitch, vel in sorted(note_map.items(), key=lambda x: x[0]):
                tok_combo.append(f'p{pitch}_v{vel}')
            
            tok_string = '-'.join(tok_combo)

            if tok_string not in self.vocab:
                self.vocab[tok_string] = len(self.vocab)
            
            tok_sequence.append(tok_string)
        
        tok_sequence.append('<eos>')

        return [self.vocab[tok] for tok in tok_sequence]
    
    def encode_all(self, midis: list[Union[str, symusic.types.Score]]) -> list[list[int]]:
        sequences = []
        for midi in tqdm.tqdm(midis, desc='Encoding Midi Files'):
            seqeunce = self.encode(midi)
            if seqeunce is not None:
                sequences.append(seqeunce)
        return sequences
    
    def decode(self, tokens: list[int] | torch.Tensor, vocab_inv: dict[int, str] = None) -> symusic.types.Score:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        if vocab_inv is None:
            vocab_inv = {v:k for k,v in self.vocab.items()}
        
        str_tokens = [vocab_inv[tok_id] for tok_id in tokens]

        toks_to_ignore = ('<pad>', '<mask>', '<bos>', '<eos>')

        track = symusic.Track('Drums', is_drum=True)
        track_pos = 0
        for tok in str_tokens:
            if tok in toks_to_ignore:
                continue

            if tok != '<rest>':
                split_notes = tok.split('-')
                for note in split_notes:
                    pitch, vel = note.split('_')
                    pitch = int(pitch[1:])
                    vel = int((int(vel[1:]) + 0.5) * (128 / self.velocity_bands)) if self.velocity_bands != 0 else 80
                    track.notes.append(symusic.Note(int(track_pos * self.tpq * (4 / self.subdivision)), 80, pitch, vel))
            
            track_pos += 1
        
        score = symusic.Score(self.tpq)
        score.tracks.append(track)

        return score

    def decode_all(self, token_set: list[list[int]] | torch.Tensor) -> list[symusic.types.Score]:
        if isinstance(token_set, torch.Tensor):
            token_set = token_set.tolist()
        
        vocab_inv = {v:k for k,v in self.vocab.items()}
        scores = []
        for tokens in token_set:
            scores.append(self.decode(tokens, vocab_inv))

        return scores

class TokeneiserII:
    def __init__(self, velocity_bands: int = 4, drum_mapping: dict[int, int] = ROLAND_DRUM_MAPPING):
        self.velocity_bands = velocity_bands
        self.drum_mapping = drum_mapping
        self.tpq = 480

        self.vocab = {
            '<pad>': 0,
            '<mask>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        self.special_token_count = len(self.vocab)

        for i in range(5):
            self.vocab[f't{i}'] = len(self.vocab)

        for pitch in drum_mapping.values():
            for vel in range(self.velocity_bands):
                key = f'p{pitch}_v{vel}'
                if key not in self.vocab:
                    self.vocab[key] = len(self.vocab)
    
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, key):
        return self.vocab[key]
        
    def encode(self, midi: Union[str, symusic.types.Score]) -> list[int] | None:
        # Load MIDI if needed
        if isinstance(midi, str):
            try:
                midi = symusic.Score.from_file(midi)
            except:
                return None
        
        if midi.tpq != self.tpq:
            midi.resample(self.tpq)
        
        drum_tracks = [track for track in midi.tracks if track.is_drum]
        
        all_notes = sorted(
            (note for track in drum_tracks for note in track.notes),
            key=lambda n: n.time
        )

        toks = []
        last_note_time = 0
        for note in all_notes:
            if note.pitch not in self.drum_mapping:
                continue

            if len(toks) != 0:
                time_delta = note.time - last_note_time
                time_tok = 4
                while time_tok >= 0:
                    while time_delta > 10 ** time_tok:
                        toks.append(f't{time_tok}')
                        time_delta -= 10 ** time_tok
                    time_tok -= 1
            
            pitch = self.drum_mapping[note.pitch]
            vel = int(note.velocity * self.velocity_bands / 128)
            toks.append(f'p{pitch}_v{vel}')
            last_note_time = note.time
        
        return [self.vocab[tok] for tok in toks]

    def encode_all(self, midis: list[Union[str, symusic.types.Score]]) -> list[list[int]]:
        sequences = []
        for midi in tqdm.tqdm(midis, desc='Encoding Midi Files'):
            sequence = self.encode(midi)
            if sequence is not None:
                sequences.append(sequence)
        return sequences
    
    def decode(self, tokens: list[int] | torch.Tensor, vocab_inv: dict[int, str] = None) -> symusic.types.Score:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        if vocab_inv is None:
            vocab_inv = {v:k for k,v in self.vocab.items()}
        
        str_tokens = [vocab_inv[tok_id] for tok_id in tokens]

        track = symusic.Track('Drums', is_drum=True)
        toks_to_ignore = ('<pad>', '<mask>', '<bos>', '<eos>')
        tick = 0

        for tok in str_tokens:
            if tok in toks_to_ignore:
                continue

            if 't' in tok:
                tick += 10 ** int(tok[-1])
            else:
                pitch, vel = tok.split('_')
                pitch = int(pitch[1:])
                vel = int((int(vel[1:]) + 0.5) * (128 / self.velocity_bands)) if self.velocity_bands != 0 else 80
                track.notes.append(symusic.Note(tick, 80, pitch, vel))
        
        score = symusic.Score(self.tpq)
        score.tracks.append(track)

        return score

    def decode_all(self, token_set: list[list[int]] | torch.Tensor) -> list[symusic.types.Score]:
        if isinstance(token_set, torch.Tensor):
            token_set = token_set.tolist()
        
        vocab_inv = {v:k for k,v in self.vocab.items()}
        scores = []
        for tokens in token_set:
            scores.append(self.decode(tokens, vocab_inv))

        return scores

class DrumSequenceEncoder:
    def __init__(self, drum_mapping: dict[int, int] = ROLAND_DRUM_MAPPING, subdivision: int = 16):
        self.drum_mapping = drum_mapping
        self.subdivision = subdivision
        self.tpq = 480
        
        self.unique_pitches = sorted(set(drum_mapping.values()))
        self.pitch_to_index = {pitch: idx for idx, pitch in enumerate(self.unique_pitches)}
        self.num_pitches = len(self.unique_pitches)
    
    def encode(self, midi: Union[str, symusic.types.Score]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load MIDI if needed
        if isinstance(midi, str):
            try:
                midi = symusic.Score.from_file(midi)
            except:
                return None
        
        drum_tracks = [track for track in midi.tracks if track.is_drum]
        if len(drum_tracks) == 0:
            return None
        
        all_notes = sorted(
            (note for track in drum_tracks for note in track.notes),
            key=lambda n: n.time
        )

        if len(all_notes) == 0:
            return None

        ticks_per_subdiv = midi.ticks_per_quarter * (4. / self.subdivision)
        
        first_note = all_notes[0].time
        quanitised_first_note = int((first_note + 0.5 * ticks_per_subdiv) / ticks_per_subdiv)
        last_note = all_notes[-1].time
        quanitised_last_note = int((last_note + 0.5 * ticks_per_subdiv) / ticks_per_subdiv)
        grid_length =  quanitised_last_note - quanitised_first_note + 1

        hit_tensor = torch.zeros(grid_length, 1, dtype=torch.float32)
        pitch_tensor = torch.zeros(grid_length, self.num_pitches, dtype=torch.float32)
        velocity_tensor = torch.zeros(grid_length, self.num_pitches, dtype=torch.float32)
        
        for note in all_notes:
            if note.pitch not in self.drum_mapping:
                continue

            grid_index = int((note.time + 0.5 * ticks_per_subdiv) / ticks_per_subdiv) - quanitised_first_note
            assert grid_index < grid_length
            
            note_pitch = self.drum_mapping[note.pitch]
            note_vel = note.velocity / 127

            pitch_index = self.pitch_to_index[note_pitch]

            hit_tensor[grid_index] = 1.0
            pitch_tensor[grid_index, pitch_index] = 1.0
            velocity_tensor[grid_index, pitch_index] = note_vel
        
        return hit_tensor, pitch_tensor, velocity_tensor
    
    def encode_all(self, midis: list[Union[str, symusic.types.Score]]) -> list[torch.Tensor]:
        sequences = []
        for midi in tqdm.tqdm(midis, desc='Encoding Midi Files'):
            sequence = self.encode(midi)
            if sequence is not None:
                sequences.append(sequence)
        return sequences
    
    def decode(self, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        hits, pitches, velocities = tensors

        assert hits.size(0) == pitches.size(0)
        assert hits.size(0) == velocities.size(0)

        track = symusic.Track('Drums', is_drum=True)
        
        inv_pitch_index = { i:p for p, i in self.pitch_to_index.items()}
        pitches = pitches.argwhere() # [grid_index, pitch_index]
        for grid_index, pitch_index in pitches:
            if hits[grid_index] == 0:
                continue
            pitch = inv_pitch_index[pitch_index.item()]
            vel = int(velocities[grid_index, pitch_index].item() * 127)
            track.notes.append(symusic.Note(int(grid_index.item() * self.tpq * 4 / self.subdivision), 80, pitch, vel))
        
        score = symusic.Score(self.tpq)
        score.tracks.append(track)

        return score
    
    def decode_all(self, sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> list[symusic.types.Score]:
        scores = []
        for sequence in sequences:
            scores.append(self.decode(sequence))

        return scores


if __name__ == '__main__':
    tokeniser = TokeneiserII(velocity_bands=4)

    midi_files = []
    for root, _, files in os.walk(os.path.join('data', 'clean_midi')):
        midi_files.extend([os.path.join(root, file) for file in files if file.endswith('.mid')]) # and 'beat' in file and 'eval' not in os.path.basename(root)])
    tok_sequences = tokeniser.encode_all(midi_files)
    print(len(tok_sequences))
    print(len(tokeniser.vocab))

    decoded = tokeniser.decode_all(tok_sequences)

    import random
    print(random.sample(decoded, k=3))


    encoder = DrumSequenceEncoder(subdivision=16)
    encoded_sequences = encoder.encode_all(midi_files)

    decoded_sequneces = encoder.decode_all(encoded_sequences)

    import random
    print(random.sample(decoded_sequneces, k=3))