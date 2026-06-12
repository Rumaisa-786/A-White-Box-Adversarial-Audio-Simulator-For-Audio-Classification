import os
import numpy as np

# === Configuration ===
TARGET_SHAPE = (64, 174)  # Expected shape (from your preprocessing)
mel_dir = os.path.abspath(os.path.join('..', 'data', 'UrbanSound8K', 'preprocessed_mels'))

print(f"📁 Checking and fixing .npy files in: {mel_dir}\n")

total = 0
fixed = 0
corrupted = 0
shapes = []

# === Helper function to pad or truncate ===
def fix_shape(mel, target_shape):
    n_mels, time_steps = mel.shape
    target_mels, target_time = target_shape

    # If too few mel bands (shouldn't happen unless preprocessing went wrong)
    if n_mels < target_mels:
        pad_height = target_mels - n_mels
        mel = np.pad(mel, ((0, pad_height), (0, 0)), mode="constant")
    elif n_mels > target_mels:
        mel = mel[:target_mels, :]

    # Fix time dimension
    if time_steps < target_time:
        pad_width = target_time - time_steps
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
    elif time_steps > target_time:
        mel = mel[:, :target_time]

    return mel


# === Main loop ===
for fname in os.listdir(mel_dir):
    if not fname.endswith('.npy'):
        continue

    fpath = os.path.join(mel_dir, fname)
    total += 1

    try:
        mel = np.load(fpath)

        if mel.shape != TARGET_SHAPE:
            mel_fixed = fix_shape(mel, TARGET_SHAPE)
            np.save(fpath, mel_fixed)
            fixed += 1
        else:
            shapes.append(mel.shape)

    except Exception as e:
        print(f"⚠️ Error reading {fname}: {e}")
        corrupted += 1


# === Summary ===
print("==========================================")
print(f"✅ Total .npy files checked: {total}")
print(f"🧩 Fixed inconsistent shapes: {fixed}")
print(f"⚠️ Corrupted/unreadable files: {corrupted}")
print(f"🎯 Final consistent shape: {TARGET_SHAPE}")
print("==========================================")

if fixed == 0 and corrupted == 0:
    print("✅ All files are already clean and ready for training!")
elif fixed > 0:
    print(f"✨ {fixed} files were automatically corrected to {TARGET_SHAPE}.")
