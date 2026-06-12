# src/train_adv.py
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import UrbanSoundDataset
from cnn_model import UrbanSoundCNN
from inaudible_attacks import masked_pgd_on_mel, compute_mask_from_mel
import numpy as np

def adversarial_train(num_epochs=10, batch_size=16, adv_eps=1.0, adv_alpha=0.25, adv_iters=4, adv_fraction=0.5, device='cpu'):
    device = torch.device(device)
    metadata_path = os.path.join('..', 'data', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    mel_folder = os.path.join('..', 'data', 'UrbanSound8K', 'preprocessed_mels')

    dataset = UrbanSoundDataset(metadata_path, mel_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    num_classes = len(dataset.label_to_idx)
    model = UrbanSoundCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("models", exist_ok=True)

    print("Starting adversarial training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start = time.time()
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.size(0)
            n_adv = int(B * adv_fraction)
            x_adv = x_batch.clone()
            if n_adv > 0:
                adv_list = []
                for i in range(n_adv):
                    mel_np = x_batch[i].detach().cpu().squeeze(0).numpy()
                    mask_np = compute_mask_from_mel(mel_np, prop=0.08)
                    xi = x_batch[i].unsqueeze(0)
                    yi = y_batch[i].unsqueeze(0)
                    xi_adv = masked_pgd_on_mel(model, xi, yi, eps=adv_eps, alpha=adv_alpha, iters=adv_iters, mask_np=mask_np, device=device)
                    adv_list.append(xi_adv.detach())
                x_adv[:n_adv] = torch.cat(adv_list, dim=0)
            x_train = torch.cat([x_adv[:n_adv], x_batch[n_adv:]], dim=0)
            y_train = torch.cat([y_batch[:n_adv], y_batch[n_adv:]], dim=0)
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        elapsed = time.time() - start
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Time: {elapsed:.1f}s")
        torch.save(model.state_dict(), os.path.join("models", "urban_sound_cnn_adv.pth"))
    print("Adversarial training finished. Model saved to models/urban_sound_cnn_adv.pth")
    return model

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--adv_frac", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    adversarial_train(num_epochs=args.epochs, batch_size=args.batch_size,
                      adv_eps=args.eps, adv_alpha=args.alpha, adv_iters=args.iters,
                      adv_fraction=args.adv_frac, device=args.device)
