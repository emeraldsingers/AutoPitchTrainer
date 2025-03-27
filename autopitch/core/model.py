import torch
import torch.nn as nn

class AutoPitchModel(nn.Module):
    def __init__(self, phoneme_vocab_size, hidden_size=512, num_layers=4, dropout=0.3):
        super(AutoPitchModel, self).__init__()
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, 128)
        self.note_fc = nn.Sequential(
            nn.Linear(7, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(256 + 128)
        self.lstm = nn.LSTM(
            input_size=256 + 128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 15)
        )

    def forward(self, note_features, phoneme_indices):
        batch_size, seq_len, _ = note_features.shape
        note_emb = self.note_fc(note_features.view(-1, 7)).view(batch_size, seq_len, -1)
        phoneme_emb = self.phoneme_embedding(phoneme_indices)
        combined = torch.cat([note_emb, phoneme_emb], dim=2)
        lstm_out, _ = self.lstm(combined)
        pitch_params = self.output_fc(lstm_out).view(batch_size, seq_len, 15)
        return pitch_params

class PitchBendLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(PitchBendLoss, self).__init__()
        self.mse = nn.SmoothL1Loss()
        self.alpha = alpha

    def forward(self, pred, target):
        base_loss = self.mse(pred, target)
        pred_deltas = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        target_deltas = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        delta_loss = self.mse(pred_deltas, target_deltas)
        return base_loss + self.alpha * delta_loss