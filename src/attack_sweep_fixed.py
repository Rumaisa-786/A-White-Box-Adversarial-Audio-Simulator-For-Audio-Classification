# attack_sweep_fixed.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from utils import UrbanSoundDataset
from cnn_model import UrbanSoundCNN
import argparse
import random
from sklearn.metrics import accuracy_score

def fgsm_attack(x, eps, data_grad, x_orig=None, clip_min=None, clip_max=None):
    """
    Single-step FGSM with optional projection into L_inf ball around x_orig
    and optional clamping to [clip_min, clip_max].
    """
    sign_grad = data_grad.sign()
    perturbed = x + eps * sign_grad
    if x_orig is not None:
        delta = torch.clamp(perturbed - x_orig, min=-eps, max=eps)
        perturbed = x_orig + delta
    if clip_min is not None and clip_max is not None:
        perturbed = torch.clamp(perturbed, clip_min, clip_max)
    return perturbed

def pgd_attack(x, y, model, eps, alpha, iters, device,
               random_start=True, clip_min=None, clip_max=None):
    """
    PGD (iterative projected gradient sign) attack with optional random start.
    x: input tensor shape (1, C, H, W) (or similar)
    y: tensor shape (1,)
    """
    x_orig = x.clone().detach().to(device)

    # random start uniformly in L_inf ball
    if random_start and eps > 0:
        rand_delta = torch.empty_like(x_orig).uniform_(-eps, eps).to(device)
        x_adv = (x_orig + rand_delta)
    else:
        x_adv = x_orig.clone().detach()

    # optionally clip to valid range before starting iterations
    if clip_min is not None and clip_max is not None:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    x_adv = x_adv.detach().requires_grad_(True)

    for _ in range(iters):
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y.to(device))

        # compute gradient w.r.t input reliably
        grads = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        if grads is None:
            break

        # step by sign of gradient
        x_adv = x_adv + alpha * grads.sign()

        # project into L_inf-ball around x_orig
        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + delta

        # optional clamp to valid input range
        if clip_min is not None and clip_max is not None:
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        # prepare for next iter
        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()

def evaluate_attack_on_samples(model, test_ds, sample_indices, eps, device,
                               pgd_params=None, clip_min=None, clip_max=None, debug=False):
    """
    Evaluate FGSM and (optional) PGD attacks on the selected samples.
    Returns (fgsm_rate, pgd_rate) where pgd_rate is None if pgd_params is None.
    """
    model.eval()
    fgsm_success = 0
    pgd_success = 0
    total = 0

    for i in sample_indices:
        x, y = test_ds[i]
        x = x.unsqueeze(0).to(device)  # shape (1, C, H, W) or similar
        y_tensor = torch.tensor([y], dtype=torch.long).to(device)

        # original prediction
        with torch.no_grad():
            pred_clean = model(x).argmax(dim=1).item()

        # FGSM: compute gradient w.r.t input
        x_fgsm = x.clone().detach().requires_grad_(True)
        out = model(x_fgsm)
        loss = F.cross_entropy(out, y_tensor)
        model.zero_grad()
        if x_fgsm.grad is not None:
            x_fgsm.grad.detach_()
            x_fgsm.grad.zero_()
        loss.backward()
        data_grad = x_fgsm.grad.data
        x_adv_fgsm = fgsm_attack(x_fgsm, eps, data_grad, x_orig=x.clone().detach(),
                                 clip_min=clip_min, clip_max=clip_max)
        with torch.no_grad():
            pred_fgsm = model(x_adv_fgsm).argmax(dim=1).item()
        if pred_fgsm != pred_clean:
            fgsm_success += 1

        # PGD
        if pgd_params is not None:
            x_adv_pgd = pgd_attack(x.clone().detach(), y_tensor, model,
                                   eps=eps,
                                   alpha=pgd_params['alpha'],
                                   iters=pgd_params['iters'],
                                   device=device,
                                   random_start=pgd_params.get('random_start', True),
                                   clip_min=clip_min,
                                   clip_max=clip_max)
            with torch.no_grad():
                pred_pgd = model(x_adv_pgd).argmax(dim=1).item()
            if pred_pgd != pred_clean:
                pgd_success += 1

            if debug and total < 5:
                # show a few debug values to help diagnose why PGD might be ineffective
                max_delta = (x_adv_pgd - x.clone().detach()).abs().max().item()
                print(f"[DEBUG] sample {i}: max_delta={max_delta:.6f}, pred_clean={pred_clean}, pred_pgd={pred_pgd}")

        total += 1

    fgsm_rate = fgsm_success / total if total > 0 else 0.0
    pgd_rate = pgd_success / total if (total > 0 and pgd_params is not None) else None
    return fgsm_rate, pgd_rate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    metadata_path = os.path.join('..', 'data', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    mel_folder = os.path.join('..', 'data', 'UrbanSound8K', 'preprocessed_mels')

    full_dataset = UrbanSoundDataset(metadata_path, mel_folder)
    total = len(full_dataset)
    test_size = int(0.2 * total)
    train_size = total - test_size

    # deterministic split
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size], generator=generator)
    print(f"Dataset total={total}, train={train_size}, test={test_size}")

    # load model
    num_classes = len(full_dataset.label_to_idx)
    model = UrbanSoundCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/urban_sound_cnn_preprocessed.pth", map_location=device))
    model.eval()

    # baseline accuracy
    def eval_acc(ds):
        preds, trues = [], []
        for x,y in DataLoader(ds, batch_size=32):
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(y.numpy().tolist())
        return accuracy_score(trues, preds)

    base_acc = eval_acc(test_ds)
    print(f"Baseline test accuracy: {base_acc*100:.2f}%")

    # Collect indices of correctly classified samples
    correct_indices = []
    for idx in range(len(test_ds)):
        x, y = test_ds[idx]
        x_t = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_t).argmax(dim=1).item()
        if pred == y:
            correct_indices.append(idx)
    print(f"Correctly classified samples (pool): {len(correct_indices)}")

    # choose subset for speed
    rng = random.Random(0)
    sample_count = min(args.samples, len(correct_indices))
    sample_indices = rng.sample(correct_indices, sample_count)
    print(f"Using {len(sample_indices)} samples for each eps.")

    eps_values = args.eps_values
    fgsm_rates = []
    pgd_rates = []

    # build pgd params (include random_start True by default)
    pgd_params = None
    if args.run_pgd:
        pgd_params = {'alpha': args.alpha, 'iters': args.iters, 'random_start': True}

    # parse clip bounds
    clip_min = args.clip_min
    clip_max = args.clip_max
    if (clip_min is None) ^ (clip_max is None):
        # if only one was provided, enforce both or none
        print("Warning: both clip_min and clip_max should be provided together. Ignoring clipping.")
        clip_min = clip_max = None

    for eps in eps_values:
        print(f"Running attacks for eps={eps} ...")
        fgsm_rate, pgd_rate = evaluate_attack_on_samples(
            model, test_ds, sample_indices, eps, device,
            pgd_params=pgd_params, clip_min=clip_min, clip_max=clip_max, debug=args.debug)
        fgsm_rates.append(fgsm_rate)
        pgd_rates.append(pgd_rate if pgd_rate is not None else 0.0)
        pgd_display = f"{(pgd_rate*100):.2f}%" if pgd_rate is not None else "N/A"
        print(f"eps={eps} => FGSM success: {fgsm_rate*100:.2f}%, PGD success: {pgd_display}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(eps_values, [r*100 for r in fgsm_rates], marker='o', label='FGSM')
    if args.run_pgd:
        plt.plot(eps_values, [r*100 for r in pgd_rates], marker='s', label=f'PGD (α={args.alpha}, iters={args.iters})')
    plt.xlabel('Epsilon (L_inf on spectrogram values)')
    plt.ylabel('Attack success rate (%)')
    plt.title('Attack success rate vs epsilon')
    plt.grid(True)
    plt.legend()
    out_path = args.out_plot
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200, help="Number of correctly-classified samples to attack per epsilon")
    parser.add_argument("--eps_values", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0], help="List of epsilons to sweep")
    parser.add_argument("--run_pgd", action="store_true", help="Also run PGD (slower)")
    parser.add_argument("--alpha", type=float, default=0.5, help="PGD step size")
    parser.add_argument("--iters", type=int, default=10, help="PGD iterations")
    parser.add_argument("--out_plot", type=str, default="../attack_success_vs_eps.png")
    parser.add_argument("--clip_min", type=float, default=None, help="Optional clip min for valid input range (e.g., 0.0)")
    parser.add_argument("--clip_max", type=float, default=None, help="Optional clip max for valid input range (e.g., 1.0)")
    parser.add_argument("--debug", action="store_true", help="Print debug info for first few samples")
    args = parser.parse_args()
    main(args)
