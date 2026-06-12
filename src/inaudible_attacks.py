# src/inaudible_attacks.py
import os
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf

def mel_db_to_linear(mel_db):
    return librosa.db_to_power(mel_db)

def compute_mask_from_mel(mel_db, prop=0.08, floor=1e-8):
    lin = mel_db_to_linear(mel_db)
    maxv = np.max(lin) if np.max(lin) > 0 else 1.0
    norm = lin / maxv
    mask = prop * norm
    mask = np.clip(mask, a_min=floor*prop, a_max=prop)
    return mask.astype(np.float32)

def spectrogram_to_waveform(mel_db, sr=22050, n_fft=1024, hop_length=512, n_iter=32):
    S = librosa.db_to_power(mel_db)
    y = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter)
    return y

def compute_snr(original_wav, adv_wav):
    eps = 1e-10
    orig = original_wav.astype(np.float32)
    adv = adv_wav.astype(np.float32)
    noise = adv - orig
    p_signal = np.sum(orig ** 2) + eps
    p_noise = np.sum(noise ** 2) + eps
    snr = 10.0 * np.log10(p_signal / p_noise + eps)
    return snr

def masked_pgd_on_mel(model, mel_tensor, label_tensor, eps, alpha, iters, mask_np, device):
    model.eval()
    x_orig = mel_tensor.clone().to(device)
    x_adv = x_orig.clone().detach().requires_grad_(True).to(device)
    mask_t = torch.from_numpy(mask_np).float().to(device).unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        out = model(x_adv)
        loss = F.cross_entropy(out, label_tensor.to(device))
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        x_adv = x_adv + alpha * grad.sign() * mask_t
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = (x_orig + delta).detach().requires_grad_(True)
    return x_adv.detach()

def generate_and_save_masked_adv_example(model, mel_db_np, true_label, out_dir,
                                         eps=1.0, alpha=0.25, iters=8, sr=22050,
                                         n_fft=1024, hop_length=512, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    model.eval()
    mel_tensor = torch.tensor(mel_db_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    label_tensor = torch.tensor([true_label], dtype=torch.long).to(device)
    mask_np = compute_mask_from_mel(mel_db_np, prop=0.08)
    adv_t = masked_pgd_on_mel(model, mel_tensor, label_tensor, eps=eps, alpha=alpha, iters=iters, mask_np=mask_np, device=device)
    adv_mel_np = adv_t.squeeze().cpu().numpy()
    orig_wav = spectrogram_to_waveform(mel_db_np, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=32)
    adv_wav = spectrogram_to_waveform(adv_mel_np, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=32)
    snr = compute_snr(orig_wav, adv_wav)
    base = f"masked_adv_eps{eps}_it{iters}"
    np.save(os.path.join(out_dir, f"{base}_orig_mel.npy"), mel_db_np)
    np.save(os.path.join(out_dir, f"{base}_adv_mel.npy"), adv_mel_np)
    sf.write(os.path.join(out_dir, f"{base}_orig.wav"), orig_wav, sr)
    sf.write(os.path.join(out_dir, f"{base}_adv.wav"), adv_wav, sr)
    return {
        "orig_wav": os.path.join(out_dir, f"{base}_orig.wav"),
        "adv_wav": os.path.join(out_dir, f"{base}_adv.wav"),
        "orig_mel": os.path.join(out_dir, f"{base}_orig_mel.npy"),
        "adv_mel": os.path.join(out_dir, f"{base}_adv_mel.npy"),
        "snr_db": float(snr)
    }
