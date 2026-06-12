# attack_demo.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from utils import UrbanSoundDataset
from cnn_model import UrbanSoundCNN
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix
import random

def fgsm_attack(x, eps, data_grad):
    # x, data_grad are tensors with shape [B,1,64,174]
    sign_grad = data_grad.sign()
    perturbed = x + eps * sign_grad
    return perturbed

def pgd_attack(x, y, model, eps, alpha, iters, device):
    # Projected Gradient Descent (untargeted)
    # x in original scale; we will keep perturbation bounded by eps (L_inf)
    x_orig = x.clone().detach()
    x_adv = x.clone().detach().requires_grad_(True).to(device)

    for i in range(iters):
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        # step
        x_adv = x_adv + alpha * grad.sign()
        # clamp to eps-ball
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = torch.clamp(x_orig + delta, min=x_orig.min().item()-10.0, max=x_orig.max().item()+10.0).detach().requires_grad_(True)
    return x_adv.detach()

def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = torch.argmax(F.softmax(out, dim=1), dim=1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.numpy().tolist())
    acc = accuracy_score(trues, preds)
    return acc, np.array(trues), np.array(preds)

def run_attacks(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Paths (adjust if your layout differs)
    metadata_path = os.path.join('..', 'data', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    mel_folder = os.path.join('..', 'data', 'UrbanSound8K', 'preprocessed_mels')

    # Load dataset
    full_dataset = UrbanSoundDataset(metadata_path, mel_folder)
    total = len(full_dataset)
    test_size = int(0.2 * total)
    train_size = total - test_size

    # deterministic split
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size], generator=generator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset: total={total}, train={train_size}, test={test_size}")

    # Load model
    num_classes = len(full_dataset.label_to_idx)
    model = UrbanSoundCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/urban_sound_cnn_preprocessed.pth", map_location=device))
    model.eval()

    # Baseline accuracy
    base_acc, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f"[Baseline] Test accuracy: {base_acc*100:.2f}%")

    # Collect a list of indices in test set that are correctly classified (we attack those)
    correct_indices = []
    model.eval()
    idx = 0
    for x,y in DataLoader(test_ds, batch_size=1, shuffle=False):
        x = x.to(device); y = y.to(device)
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(dim=1)
        if pred.item() == y.item():
            correct_indices.append(idx)
        idx += 1
    print(f"Correctly classified test samples: {len(correct_indices)} / {len(test_ds)}")

    # FGSM attack (untargeted)
    eps = args.eps  # epsilon in same units as mel tensor (experimentally pick small value)
    adv_success = 0
    adv_total = 0

    # We'll run FGSM on a random subset of correctly classified examples (for speed)
    sample_indices = correct_indices if args.samples is None else random.sample(correct_indices, min(args.samples, len(correct_indices)))
    print(f"Running FGSM on {len(sample_indices)} samples (eps={eps}) ...")

    for i in sample_indices:
        x, y = test_ds[i]
        x = x.unsqueeze(0).to(device).requires_grad_(True)  # [1,1,64,174]
        y = torch.tensor([y]).to(device)

        out = model(x)
        loss = F.cross_entropy(out, y)
        model.zero_grad()
        loss.backward()
        data_grad = x.grad.data

        x_adv = fgsm_attack(x, eps, data_grad)
        # optional clamp: keep spectrograms in reasonable range — here we don't know exact min/max; leave as-is
        with torch.no_grad():
            out_adv = model(x_adv)
            pred_adv = out_adv.argmax(dim=1).item()
            pred_clean = out.argmax(dim=1).item()

        adv_total += 1
        if pred_adv != pred_clean:
            adv_success += 1

        # save a few examples
        if adv_total <= 5:
            save_example(x.detach().cpu().squeeze().numpy(), x_adv.detach().cpu().squeeze().numpy(),
                         y.item(), pred_clean, pred_adv, i, args.out_dir)

    print(f"FGSM: success {adv_success}/{adv_total} -> success rate {(adv_success/adv_total*100):.2f}%")

    # PGD attack
    print(f"\nRunning PGD (eps={eps}, alpha={args.alpha}, iters={args.iters}) on {len(sample_indices)} samples ...")
    adv_success_pgd = 0
    adv_total_pgd = 0
    for i in sample_indices:
        x, y = test_ds[i]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([y]).to(device)

        x_adv = pgd_attack(x, y, model, eps=eps, alpha=args.alpha, iters=args.iters, device=device)
        with torch.no_grad():
            out_clean = model(x)
            out_adv = model(x_adv)
            pred_clean = out_clean.argmax(dim=1).item()
            pred_adv = out_adv.argmax(dim=1).item()

        adv_total_pgd += 1
        if pred_adv != pred_clean:
            adv_success_pgd += 1

        if adv_total_pgd <= 5:
            save_example(x.detach().cpu().squeeze().numpy(), x_adv.detach().cpu().squeeze().numpy(),
                         y.item(), pred_clean, pred_adv, i, args.out_dir, prefix="pgd")

    print(f"PGD: success {adv_success_pgd}/{adv_total_pgd} -> success rate {(adv_success_pgd/adv_total_pgd*100):.2f}%")

def save_example(orig_np, adv_np, true_label, pred_clean, pred_adv, idx, out_dir, prefix="fgsm"):
    os.makedirs(out_dir, exist_ok=True)
    # save arrays
    np.save(os.path.join(out_dir, f"{prefix}_orig_{idx}.npy"), orig_np)
    np.save(os.path.join(out_dir, f"{prefix}_adv_{idx}.npy"), adv_np)
    # plot and save image
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.imshow(orig_np, aspect='auto', origin='lower')
    plt.title(f"orig true={true_label} pred={pred_clean}")
    plt.colorbar(format="%+2.0f dB")
    plt.subplot(1,2,2)
    plt.imshow(adv_np, aspect='auto', origin='lower')
    plt.title(f"adv pred={pred_adv}")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_compare_{idx}.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eps", type=float, default=2.0, help="FGSM/PGD epsilon (L_inf) on mel spectrogram values")
    parser.add_argument("--alpha", type=float, default=0.5, help="PGD step size")
    parser.add_argument("--iters", type=int, default=10, help="PGD iterations")
    parser.add_argument("--samples", type=int, default=200, help="Number of correctly-classified test samples to attack (None=all)")
    parser.add_argument("--out_dir", type=str, default="../attack_outputs", help="where to save examples")
    args = parser.parse_args()
    run_attacks(args)
