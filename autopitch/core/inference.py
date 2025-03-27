import torch
import yaml
import os
import numpy as np
from .data_utils import prepare_batch, _extract_note_feature
from .model import AutoPitchModel 
from ..utils import device

def load_model_with_vocab(
    model_path: str, 
    phoneme_to_idx: dict
) -> AutoPitchModel:
    """
    Loads a model from a saved checkpoint and updates the vocabulary size.
    
    Args:
        model_path: The path to the saved model checkpoint.
        phoneme_to_idx: The current phoneme-to-index mapping for the model.
    
    Returns:
        The loaded model with the updated vocabulary size.
    """
    checkpoint = torch.load(model_path, map_location=device)
    saved_vocab_size = checkpoint.get('phoneme_vocab_size', len(phoneme_to_idx))
    model = AutoPitchModel(
        phoneme_vocab_size=len(phoneme_to_idx),
        hidden_size=512,
        num_layers=4,
        dropout=0.3
    ).to(device)
    state_dict = checkpoint['model_state_dict']
    if saved_vocab_size != len(phoneme_to_idx):
        print(f"Vocabulary size changed from {saved_vocab_size} to {len(phoneme_to_idx)}")
        if 'phoneme_embedding.weight' in state_dict:
            del state_dict['phoneme_embedding.weight']
    model.load_state_dict(state_dict, strict=False)
    return model

def process_ustx_file(
    model: AutoPitchModel,
    ustx_path: str,
    output_path: str,
    phoneme_to_idx: dict
) -> None:
    """
    Processes a ustx file and save the output to the specified path.
    
    Args:
        model: The model to use for inference.
        ustx_path: The path to the ustx file to process.
        output_path: The path to save the processed ustx file.
        phoneme_to_idx: The current phoneme-to-index mapping for the model.
    """
    with open(ustx_path, 'r', encoding='utf-8') as f:
        ustx_data = yaml.safe_load(f)

    model.eval()

    for part_idx, part in enumerate(ustx_data.get('voice_parts', [])):
        notes = part.get('notes', [])
        part_features = []
        part_notes_indices = []

        for note_idx, note in enumerate(notes):
            feature = _extract_note_feature(notes, note_idx, phoneme_to_idx)
            if feature:
                part_features.append(feature)
                part_notes_indices.append(note_idx)

        if not part_features:
            continue

        note_features_tensor, phoneme_indices_tensor = prepare_batch([part_features])

        if note_features_tensor is None:
             continue

        with torch.no_grad():
            predicted_params_seq = model(note_features_tensor, phoneme_indices_tensor)

        predicted_params = predicted_params_seq.cpu().numpy()[0]

        for i, note_original_idx in enumerate(part_notes_indices):
            note = notes[note_original_idx]
            params = predicted_params[i]
            pitch_data = []
            last_x = 0
            shape_map = ['io', 'i', 'o', 'l', 'li', 'lo'] 
            for j in range(5):
                x, y, shape_value = params[j*3:(j+1)*3]
                if abs(x) < 1e-6 and abs(y) < 1e-6 and abs(shape_value) < 1e-6:
                    continue
                x = float(x * 100)
                y = float(y * 100)
                shape_idx = min(max(0, round(shape_value * 5)), 5)
                shape_str = shape_map[shape_idx]
                pitch_data.append({'x': x, 'y': y, 'shape': shape_str})
                last_x = x

            pitch_data.append({'x': last_x+15, 'y': 0, 'shape': 'io'})

            if 'pitch' not in note:
                note['pitch'] = {}
            note['pitch']['data'] = pitch_data
            note['pitch']['snap_first'] = True

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('name: Merged Project\n')
        f.write('comment: "Tuned by Autopitch made by asoqwer (Emerald Project)"\n')
        f.write('output_dir: Vocal\n')
        f.write('cache_dir: UCache\n')
        f.write('ustx_version: "0.6"\n')
        f.write('resolution: 480\n')
        f.write('bpm: 120\n')
        f.write('beat_per_bar: 4\n')
        yaml.dump(ustx_data, f, allow_unicode=True)

    print(f"Processed {ustx_path} and saved to {output_path}")
