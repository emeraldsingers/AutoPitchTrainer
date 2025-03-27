import torch
import os
import shutil
from .model import AutoPitchModel
from ..utils import log_queue # log_queue не используется, но импорт оставлен
from ..config import ONNX_OPSET_VERSION

def convert_to_onnx(model_path, output_onnx_path, phoneme_vocab_size):
    checkpoint_path = model_path
    vocab_yaml_path = model_path.replace('last_autopitch_model.pth', 'phoneme_vocab.yaml')
    output_dir = os.path.dirname(output_onnx_path)
    output_vocab_yaml_path = os.path.join(output_dir, 'phoneme_vocab.yaml')


    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = AutoPitchModel(
        phoneme_vocab_size=phoneme_vocab_size,
        hidden_size=512,
        num_layers=3,
        dropout=0.3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    batch_size = 1
    seq_length = 25
    note_features = torch.zeros((batch_size, seq_length, 7))
    phoneme_indices = torch.zeros((batch_size, seq_length), dtype=torch.long)

    try:
        os.makedirs(output_dir, exist_ok=True)
        torch.onnx.export(
            model,
            (note_features, phoneme_indices),
            output_onnx_path,
            export_params=True,
            opset_version=ONNX_OPSET_VERSION,
            do_constant_folding=True,
            input_names=['note_features', 'phoneme_indices'],
            output_names=['pitch_params'],
            dynamic_axes={
                'note_features': {0: 'batch_size', 1: 'seq_length'},
                'phoneme_indices': {0: 'batch_size', 1: 'seq_length'},
                'pitch_params': {0: 'batch_size', 1: 'seq_length'}
            }
        )
        shutil.copyfile(vocab_yaml_path, output_vocab_yaml_path)
        print(f"Model converted to ONNX: {output_onnx_path}")

    except Exception as e:
        print(f"Error while converting model to ONNX: {str(e)}")

def frequency_changes(y_values):
    diffs = torch.abs(y_values[:, :, 1:] - y_values[:, :, :-1])
    return diffs.sum(dim=-1).mean()