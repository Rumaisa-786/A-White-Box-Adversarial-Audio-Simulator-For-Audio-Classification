import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ==== Paths ====
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'UrbanSound8K'))
metadata_path = os.path.join(base_dir, 'metadata', 'UrbanSound8K.csv')
audio_folder = os.path.join(base_dir, 'audio')
output_folder = os.path.join(base_dir, 'preprocessed_mels')

os.makedirs(output_folder, exist_ok=True)
print(f"📁 Saving processed files to: {output_folder}\n")

# ==== Load metadata ====
metadata = pd.read_csv(metadata_path)

# ==== Parameters ====
sr = 22050
n_mels = 64
n_fft = 1024
hop_length = 512
max_len = 174  # fixed time dimension (you can tweak this)

# ==== Process each audio file ====
for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
    fold = row['fold']
    filename = row['slice_file_name']
    label = row['class']

    input_path = os.path.join(audio_folder, f"fold{fold}", filename)
    output_path = os.path.join(output_folder, filename.replace('.wav', '.npy'))

    # Skip if already processed
    if os.path.exists(output_path):
        continue

    try:
        # Load and compute mel spectrogram
        y, sr = librosa.load(input_path, sr=sr)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad or truncate to fixed length
        if mel_db.shape[1] < max_len:
            pad_width = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]

        # ✅ Save as .npy file
        np.save(output_path, mel_db)
        print(f"💾 Saved: {output_path}")

    except Exception as e:
        print(f"⚠️ Error processing {input_path}: {e}")

print("\n✅ Preprocessing complete! All mel spectrograms saved to:", output_folder)
