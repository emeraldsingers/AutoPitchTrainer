import torch
import numpy as np
import yaml
import os
import re
from concurrent.futures import ThreadPoolExecutor
from ..utils import device
from ..config import MAX_SEQ_LENGTH, NUM_THREADS

def load_ustx_file(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            print(f"Загрузка {file}")
            ustx_data = yaml.safe_load(f)
        return ustx_data
    except Exception as e:
        print(f"Ошибка при загрузке {file}: {e}")
        return None

def _extract_note_feature(notes, note_idx, phoneme_to_idx):
    note = notes[note_idx]
    note_features = [0.0] * 7
    note_features[0] = note.get('duration', 0) / 480

    if note_idx > 0:
        prev_note = notes[note_idx - 1]
        note_features[1] = (note['tone'] - prev_note.get('tone', 60)) / 127
        note_features[5] = 1.0
    else:
        note_features[1] = 0.0
        note_features[5] = 0.0

    if note_idx > 0:
        prev_note = notes[note_idx - 1]
        rest_length = note['position'] - (prev_note['position'] + prev_note['duration'])
        note_features[3] = 1.0 if rest_length > 0 else 0.0
    else:
        note_features[3] = 0.0

    if note_idx < len(notes) - 1:
        next_note = notes[note_idx + 1]
        note_features[2] = (next_note.get('tone', 60) - note['tone']) / 127
        note_features[6] = 1.0
    else:
        note_features[2] = 0.0
        note_features[6] = 0.0

    phoneme = str(note.get('lyric', '<UNK>'))
    if " " in phoneme:
        phoneme = phoneme.split(" ")[1]

    if phoneme not in ["息", "吸", '・', "R"]:
        phoneme = re.sub(r"[A-QS-Z0-9・ '\＃↑裏_]", "", phoneme, flags=re.UNICODE)

    phoneme_idx = phoneme_to_idx.get(phoneme, phoneme_to_idx['<UNK>'])
    return (note_features, phoneme_idx)

def _extract_pitch_parameters(note):
    pitch_data = note['pitch']['data']
    params = []
    for point in pitch_data[:5]:
        if point.get('x', 0) in [-40, 40] and point.get('y', 0) in [-40, 40, 0]:
            continue
        x = point.get('x', 0) / 100
        y = point.get('y', 0) / 100
        shape_map = {'io': 0, 'i': 1, 'o': 2, 'l': 3, 'li': 4, 'lo': 5}
        shape_value = shape_map.get(point.get('shape', 'io'), 0) / 5
        params.extend([x, y, shape_value])
    while len(params) < 15:
        params.extend([0, 0, 0])
    return params

def load_and_preprocess_ustx_files(ustx_files, max_sequence_length=MAX_SEQ_LENGTH, num_threads=NUM_THREADS):
    all_phonemes = set(['<PAD>', '<UNK>'])
    all_data = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(load_ustx_file, file) for file in ustx_files]
        for future in futures:
            ustx_data = future.result()
            if ustx_data:
                all_data.append(ustx_data)

    all_features = []
    all_labels = []

    for ustx_data in all_data:
        for part in ustx_data.get('voice_parts', []):
            for note in part.get('notes', []):
                phoneme = str(note.get('lyric', '<UNK>'))
                if " " in phoneme:
                    phoneme = phoneme.split(" ")[1]
                if phoneme not in ["息", "吸", '・', "R"]:
                    phoneme = re.sub(r"[A-QS-Z0-9・ '\＃\↑裏_]", "", phoneme, flags=re.UNICODE)
                all_phonemes.add(phoneme)

    phoneme_list = sorted(list(all_phonemes))
    phoneme_to_idx = {p: i for i, p in enumerate(phoneme_list)}
    idx_to_phoneme = {i: p for p, i in phoneme_to_idx.items()}

    for ustx_data in all_data:
        for part in ustx_data.get('voice_parts', []):
            notes = part.get('notes', [])
            note_features = []
            note_labels = []
            for note_idx, note in enumerate(notes):
                if 'pitch' not in note or not note['pitch'].get('data'):
                    continue
                feature = _extract_note_feature(notes, note_idx, phoneme_to_idx)
                label = _extract_pitch_parameters(note)
                if feature and label:
                    note_features.append(feature)
                    note_labels.append(label)
            if len(note_features) > 0:
                if len(note_features) > max_sequence_length:
                    for i in range(0, len(note_features), max_sequence_length):
                        all_features.append(note_features[i:i+max_sequence_length])
                        all_labels.append(note_labels[i:i+max_sequence_length])
                else:
                    all_features.append(note_features)
                    all_labels.append(note_labels)

    return all_features, all_labels, phoneme_to_idx, idx_to_phoneme


def prepare_batch(features_list, labels_list=None):
    batch_size = len(features_list)
    max_seq_length = max(len(features) for features in features_list)
    note_features = np.zeros((batch_size, max_seq_length, 7), dtype=np.float32)
    phoneme_indices = np.zeros((batch_size, max_seq_length), dtype=np.int64)

    for i, seq_features in enumerate(features_list):
        for j, (note_feature, phoneme_idx) in enumerate(seq_features):
            note_features[i, j] = note_feature
            phoneme_indices[i, j] = phoneme_idx

    note_features_tensor = torch.from_numpy(note_features).to(device)
    phoneme_indices_tensor = torch.from_numpy(phoneme_indices).to(device)

    if labels_list:
        labels = np.zeros((batch_size, max_seq_length, 15), dtype=np.float32)
        for i, seq_labels in enumerate(labels_list):
            for j, label in enumerate(seq_labels):
                labels[i, j] = label
        labels_tensor = torch.from_numpy(labels).to(device)
        return note_features_tensor, phoneme_indices_tensor, labels_tensor
    return note_features_tensor, phoneme_indices_tensor