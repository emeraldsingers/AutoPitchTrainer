import yaml
import os
from typing import Dict, Tuple

def save_vocabulary(vocab_path: str, phoneme_to_idx: Dict[str, int], idx_to_phoneme: Dict[int, str]) -> None:
    """
    Saves the vocabulary to a YAML file.

    Args:
        vocab_path (str): The path to the file to save the vocabulary to.
        phoneme_to_idx (Dict[str, int]): The mapping of phonemes to indices.
        idx_to_phoneme (Dict[int, str]): The mapping of indices to phonemes.
    """
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'phoneme_to_idx': phoneme_to_idx,
            'idx_to_phoneme': idx_to_phoneme
        }, f, allow_unicode=True)
    print(f"Vocabulary saved to {vocab_path}")

def load_vocabulary(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Loads the vocabulary from a YAML file.

    Args:
        vocab_path (str): The path to the file to load the vocabulary from.

    Returns:
        phoneme_to_idx (Dict[str, int]): The mapping of phonemes to indices.
        idx_to_phoneme (Dict[int, str]): The mapping of indices to phonemes.
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = yaml.safe_load(f)
    return vocab_data['phoneme_to_idx'], vocab_data['idx_to_phoneme']
