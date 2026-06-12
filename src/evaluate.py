import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from cnn_model import UrbanSoundCNN
from utils import UrbanSoundDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# ==================================
# 🔹 Evaluate a given model
# ==================================
def evaluate_model(model_path, dataset, device, idx_to_label, model_name="Model"):
    print(f"\n🔹 Evaluating {model_name} ...")
    num_classes = len(idx_to_label)
    model = UrbanSoundCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for mels, labels in test_loader:
            mels, labels = mels.to(device), labels.to(device)
            outputs = model(mels)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ {model_name} Accuracy: {acc * 100:.2f}%")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[idx_to_label[i] for i in range(num_classes)],
        digits=3
    )
    print(f"\n📊 {model_name} Classification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[idx_to_label[i] for i in range(num_classes)],
        yticklabels=[idx_to_label[i] for i in range(num_classes)]
    )
    plt.title(f"{model_name} - Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return acc


# ==================================
# 🔹 MAIN
# ==================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    metadata_path = os.path.join('..', 'data', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    mel_folder = os.path.join('..', 'data', 'UrbanSound8K', 'preprocessed_mels')

    print("🔄 Loading dataset...")
    full_dataset = UrbanSoundDataset(metadata_path, mel_folder)
    total_size = len(full_dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f"✅ Dataset loaded: {total_size} samples ({train_size} train / {test_size} test)")
    idx_to_label = {v: k for k, v in full_dataset.label_to_idx.items()}

    # Paths to both models
    baseline_model = "models/urban_sound_cnn_preprocessed.pth"
    adv_model = "models/urban_sound_cnn_adv.pth"

    # Evaluate both
    acc_base = evaluate_model(baseline_model, test_dataset, device, idx_to_label, "Baseline CNN")
    acc_adv = evaluate_model(adv_model, test_dataset, device, idx_to_label, "Adversarially Trained CNN")

    # Compare visually
    plt.figure(figsize=(6, 4))
    plt.bar(["Baseline CNN", "Adversarial CNN"], [acc_base * 100, acc_adv * 100], color=["skyblue", "orange"])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison: Baseline vs Adversarially Trained")
    plt.tight_layout()
    plt.show()
