import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class UrbanSoundDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(self.metadata['class'].unique())}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['slice_file_name'].replace('.wav', '.npy')
        label = self.label_to_idx[row['class']]

        # ✅ Load precomputed mel spectrogram
        mel_path = os.path.join(self.audio_dir, filename)
        mel = np.load(mel_path)  # shape: (n_mels, time_steps)

        # Add channel dimension for CNN: (1, n_mels, time_steps)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        return mel_tensor, torch.tensor(label, dtype=torch.long)
