import yaml
import os

def save_vocabulary(vocab_path, phoneme_to_idx, idx_to_phoneme):
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'phoneme_to_idx': phoneme_to_idx,
            'idx_to_phoneme': idx_to_phoneme
        }, f, allow_unicode=True)
    print(f"Vocabulary saved to {vocab_path}")

def load_vocabulary(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = yaml.safe_load(f)
    return vocab_data['phoneme_to_idx'], vocab_data['idx_to_phoneme']