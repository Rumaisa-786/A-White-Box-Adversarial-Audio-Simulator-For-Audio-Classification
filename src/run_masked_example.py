import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from cnn_model import UrbanSoundCNN

# ---------------------------
# Utility Functions
# ---------------------------

def find_wav_for_mel(mel_path, audio_root="../data/UrbanSound8K/audio"):
    base = os.path.splitext(os.path.basename(mel_path))[0]
    pattern = os.path.join(audio_root, "**", base + ".wav")
    hits = glob.glob(pattern, recursive=True)
    if len(hits) == 0:
        raise FileNotFoundError(f"No wav found for {base} under {audio_root}")
    return hits[0]

def get_mel_tensor(waveform, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Convert waveform tensor to mel spectrogram in dB scale."""
    import torchaudio  # only needed here, safe to keep
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    amptodb = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
    mel = mel_transform(waveform)
    mel_db = amptodb(mel)
    return mel_db

# ---------------------------
# PGD Attack
# ---------------------------

def pgd_waveform_attack(model, orig_wav, label, sr,
                        eps=0.05, alpha=0.01, iters=8,
                        device='cpu', debug=True):
    """Projected Gradient Descent attack on waveform input."""
    model = model.to(device)
    model.eval()

    adv_wav = orig_wav.clone().to(device)
    orig_wav = orig_wav.to(device)
    label = label.to(device)

    for i in range(iters):
        adv_wav.requires_grad = True

        mel = get_mel_tensor(adv_wav, sr)
        inp = mel.unsqueeze(0).to(device).float()  # [batch, 1, n_mels, time]
        logits = model(inp)
        loss = F.cross_entropy(logits, label)

        model.zero_grad()
        if adv_wav.grad is not None:
            adv_wav.grad.zero_()
        loss.backward()

        grad_sign = adv_wav.grad.data.sign()
        adv_wav = adv_wav.data + alpha * grad_sign
        delta = torch.clamp(adv_wav - orig_wav, min=-eps, max=eps)
        adv_wav = torch.clamp(orig_wav + delta, min=-1.0, max=1.0).detach()

        if debug:
            mean_pert = torch.mean(torch.abs(adv_wav - orig_wav)).item()
            print(f"Iter {i+1}/{iters}: loss={loss.item():.4f}, mean_pert={mean_pert:.6f}")

    return adv_wav.detach().cpu()

# ---------------------------
# Safe WAV saving (no FFmpeg needed)
# ---------------------------

def save_wav(tensor_wav, sr, out_path):
    """Save waveform tensor as a .wav using soundfile."""
    if isinstance(tensor_wav, torch.Tensor):
        wav_np = tensor_wav.detach().cpu().numpy().squeeze()
    else:
        wav_np = np.asarray(tensor_wav).squeeze()
    wav_np = np.clip(wav_np, -1.0, 1.0)
    sf.write(out_path, wav_np, sr)
    print(f"✅ Saved: {out_path}")

# ---------------------------
# Main Execution
# ---------------------------

def main():
    mel_path = "../data/UrbanSound8K/preprocessed_mels/100032-3-0-0.npy"
    out_dir = "../attack_outputs_masked"
    os.makedirs(out_dir, exist_ok=True)

    sr = 22050
    eps = 0.05
    alpha = 0.01
    iters = 8
    true_label = 3
    device = 'cpu'

    num_classes = 10
    model = UrbanSoundCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("models/urban_sound_cnn_preprocessed.pth", map_location=device))
    model.eval()

    wav_path = find_wav_for_mel(mel_path, audio_root="../data/UrbanSound8K/audio")
    print("🎵 Found wav:", wav_path)

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    orig_wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    print(f"Original wav shape: {orig_wav.shape}, min={orig_wav.min().item():.4f}, max={orig_wav.max().item():.4f}")

    label_tensor = torch.LongTensor([true_label])
    adv_wav = pgd_waveform_attack(model, orig_wav, label_tensor, sr,
                                  eps=eps, alpha=alpha, iters=iters,
                                  device=device, debug=True)

    base = os.path.splitext(os.path.basename(mel_path))[0]
    orig_out = os.path.join(out_dir, f"{base}_orig.wav")
    adv_out  = os.path.join(out_dir, f"{base}_adv.wav")
    diff_out = os.path.join(out_dir, f"{base}_perturbation.wav")

    # Save all
    save_wav(orig_wav, sr, orig_out)
    save_wav(adv_wav, sr, adv_out)
    save_wav(adv_wav - orig_wav, sr, diff_out)

    # quick verification
    mean_abs_diff = torch.mean(torch.abs(adv_wav - orig_wav)).item()
    print(f"🔍 Mean absolute perturbation: {mean_abs_diff:.6f}")

    print(f"\n✅ Done! Files saved in: {os.path.abspath(out_dir)}")
    print(f" - Original : {orig_out}")
    print(f" - Adversarial : {adv_out}")
    print(f" - Perturbation : {diff_out}")

if __name__ == "__main__":
    main()
    
