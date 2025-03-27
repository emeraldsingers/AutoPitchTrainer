import torch
import torch.optim as optim
import numpy as np
import os
from .model import PitchBendLoss
from .data_utils import prepare_batch
from ..utils import device
from ..config import WEIGHT_DECAY, PITCH_BEND_LOSS_ALPHA, GRAD_CLIP_NORM

def train_model(model, train_data, val_data, phoneme_to_idx, config, output_dir, resume=False, freeze_layers=False):
    train_features, train_labels = train_data
    val_features, val_labels = val_data
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    loss_fn = PitchBendLoss(alpha=PITCH_BEND_LOSS_ALPHA)

    print(config)
    start_epoch = 0
    if resume:
        checkpoint = torch.load(os.path.join(output_dir, 'last_autopitch_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    if freeze_layers:
        for name, param in model.named_parameters():
            if 'lstm' in name and '0' in name:
                param.requires_grad = False

    best_val_loss = float('inf')
    print(f"Starting training from epoch {start_epoch}. Total data: {len(train_features)}, validation data: {len(val_features)}")

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        total_train_loss = 0

        combined = list(zip(train_features, train_labels))
        np.random.shuffle(combined)
        shuffled_features, shuffled_labels = zip(*combined) if combined else ([], [])

        for i in range(0, len(shuffled_features), config['batch_size']):
            batch_features = list(shuffled_features[i:i+config['batch_size']])
            batch_labels = list(shuffled_labels[i:i+config['batch_size']])

            if len(batch_features) < 2:
                continue

            note_features_tensor, phoneme_indices_tensor, labels_tensor = prepare_batch(batch_features, batch_labels)

            if note_features_tensor is None: 
                 continue

            optimizer.zero_grad()
            outputs = model(note_features_tensor, phoneme_indices_tensor)
            loss = loss_fn(outputs, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_features), config['batch_size']):
                batch_features = val_features[i:i+config['batch_size']]
                batch_labels = val_labels[i:i+config['batch_size']]

                if len(batch_features) < 2:
                    continue

                note_features_tensor, phoneme_indices_tensor, labels_tensor = prepare_batch(batch_features, batch_labels)

                if note_features_tensor is None: 
                    continue

                outputs = model(note_features_tensor, phoneme_indices_tensor)
                loss = loss_fn(outputs, labels_tensor)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_features) if len(train_features) > 0 else 0
        avg_val_loss = total_val_loss / len(val_features) if len(val_features) > 0 else 0
        scheduler.step()

        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")

        if epoch % 50 == 0 or epoch == config['epochs'] - 1:
            model_path = os.path.join(output_dir, 'last_autopitch_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'phoneme_vocab_size': len(phoneme_to_idx)
            }, model_path)
            print(f"Saved model to {model_path} from epoch {epoch}")

    return model