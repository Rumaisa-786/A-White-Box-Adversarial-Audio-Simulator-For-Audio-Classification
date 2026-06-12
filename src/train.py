import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from utils import UrbanSoundDataset
from cnn_model import UrbanSoundCNN
import os
import time

if __name__ == "__main__":   # ✅ <-- ADD THIS LINE
    # =============================
    # 🔹 PATHS
    # =============================
    metadata_path = os.path.join('..', 'data', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    mel_folder = os.path.join('..', 'data', 'UrbanSound8K', 'preprocessed_mels')

    # =============================
    # 🔹 DATASET & DATALOADER
    # =============================
    print("🔄 Loading dataset...")
    dataset = UrbanSoundDataset(metadata_path, mel_folder)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,    # ✅ use 0 workers for safety on Windows
        pin_memory=True
    )

    print(f"✅ Loaded {len(dataset)} samples")

    # =============================
    # 🔹 MODEL SETUP
    # =============================
    num_classes = len(dataset.label_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    model = UrbanSoundCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # =============================
    # 🔹 TRAINING LOOP
    # =============================
    num_epochs = 5
    print("🚀 Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for mel_specs, labels in dataloader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"📘 Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

    print("✅ Training complete!")

    # =============================
    # 🔹 SAVE MODEL
    # =============================
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "urban_sound_cnn_preprocessed.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")
 