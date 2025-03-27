import os
import torch
from .core.data_utils import load_and_preprocess_ustx_files
from .core.vocabulary import save_vocabulary, load_vocabulary
from .core.model import AutoPitchModel
from .core.training import train_model
from .core.inference import process_ustx_file, load_model_with_vocab
from .core.onnx_converter import convert_to_onnx
from .utils import device, log_queue
from .config import (
    DEFAULT_LR, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT
)

def run_training(data_dir, output_dir, epochs, batch_size, q):
    q.put("Starting training...")
    ustx_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.ustx')]
    if not ustx_files:
        q.put("No USTX files found in the data directory.")
        return
    all_features, all_labels, phoneme_to_idx, idx_to_phoneme = load_and_preprocess_ustx_files(ustx_files)
    vocab_path = os.path.join(output_dir, 'phoneme_vocab.yaml')
    save_vocabulary(vocab_path, phoneme_to_idx, idx_to_phoneme)
    split = int(0.8 * len(all_features))
    train_data = (all_features[:split], all_labels[:split])
    val_data = (all_features[split:], all_labels[split:])
    config = {'batch_size': batch_size, 'lr': DEFAULT_LR, 'epochs': epochs, 'hidden_size': DEFAULT_HIDDEN_SIZE, 'num_layers': DEFAULT_NUM_LAYERS, 'dropout': DEFAULT_DROPOUT}
    model = AutoPitchModel(phoneme_vocab_size=len(phoneme_to_idx), hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config['dropout']).to(device)
    train_model(model, train_data, val_data, phoneme_to_idx, config, output_dir)
    q.put("Training completed.")

def run_fine_tuning(model_dir, data_dir, output_dir, epochs, batch_size, freeze_layers, q):
    q.put("Starting fine-tuning...")
    ustx_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.ustx')]
    if not ustx_files:
        q.put("No USTX files found in the data directory.")
        return
    all_features, all_labels, phoneme_to_idx, idx_to_phoneme = load_and_preprocess_ustx_files(ustx_files)
    vocab_path = os.path.join(output_dir, 'phoneme_vocab.yaml')
    save_vocabulary(vocab_path, phoneme_to_idx, idx_to_phoneme)
    split = int(0.8 * len(all_features))
    train_data = (all_features[:split], all_labels[:split])
    val_data = (all_features[split:], all_labels[split:])
    config = {'batch_size': batch_size, 'lr': DEFAULT_LR, 'epochs': epochs, 'hidden_size': DEFAULT_HIDDEN_SIZE, 'num_layers': DEFAULT_NUM_LAYERS, 'dropout': DEFAULT_DROPOUT}
    model = AutoPitchModel(phoneme_vocab_size=len(phoneme_to_idx), hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config['dropout']).to(device)
    checkpoint = torch.load(os.path.join(model_dir, 'last_autopitch_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    train_model(model, train_data, val_data, phoneme_to_idx, config, output_dir, resume=False, freeze_layers=freeze_layers)
    q.put("Fine-tuning completed.")

def run_resume_training(model_dir, data_dir, output_dir, epochs, batch_size, q):
    q.put("Starting resume training...")
    ustx_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.ustx')]
    if not ustx_files:
        q.put("No USTX files found in the data directory.")
        return
    all_features, all_labels, phoneme_to_idx, idx_to_phoneme = load_and_preprocess_ustx_files(ustx_files)
    vocab_path = os.path.join(output_dir, 'phoneme_vocab.yaml')
    save_vocabulary(vocab_path, phoneme_to_idx, idx_to_phoneme)
    split = int(0.8 * len(all_features))
    train_data = (all_features[:split], all_labels[:split])
    val_data = (all_features[split:], all_labels[split:])
    config = {'batch_size': batch_size, 'lr': DEFAULT_LR, 'epochs': epochs, 'hidden_size': DEFAULT_HIDDEN_SIZE, 'num_layers': DEFAULT_NUM_LAYERS, 'dropout': DEFAULT_DROPOUT}
    model = AutoPitchModel(phoneme_vocab_size=len(phoneme_to_idx), hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config['dropout']).to(device)
    train_model(model, train_data, val_data, phoneme_to_idx, config, output_dir, resume=True)
    q.put("Resume training completed.")

def run_processing(model_file, input_ustx, output_dir, vocab_file, q):
    q.put("Starting processing...")
    phoneme_to_idx, _ = load_vocabulary(vocab_file)
    model = load_model_with_vocab(model_file, phoneme_to_idx)
    model.eval()
    if os.path.isfile(input_ustx):
        input_files = [input_ustx]
        output_files = [os.path.join(output_dir, f"autotuned_{os.path.basename(input_ustx)}")]
    else:
        input_files = [os.path.join(root, f) for root, _, files in os.walk(input_ustx) for f in files if f.endswith('.ustx')]
        output_files = [os.path.join(output_dir, f"autotuned_{os.path.basename(f)}") for f in input_files]

    os.makedirs(output_dir, exist_ok=True)

    for input_file, output_file in zip(input_files, output_files):
        q.put(f"Processing {os.path.basename(input_file)}...")
        process_ustx_file(model, input_file, output_file, phoneme_to_idx)
    q.put("Processing completed.")

def run_conversion(model_path, output_onnx_path, phoneme_vocab_size, q):
    q.put("Starting ONNX conversion...")
    convert_to_onnx(model_path, output_onnx_path, phoneme_vocab_size)
    q.put(f"Model conversion attempt finished for {output_onnx_path}") 