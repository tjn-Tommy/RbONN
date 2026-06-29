"""Stage 2-3: Train RbONNTwin in pure simulation with real-time Trackio monitoring.

Architecture
------------
The SLM cycles through 10 weight rows (one per digit class), each with 20 phases.
The scope records one 420 nm intensity per row (10 time windows).

    phi_w in R^{10 x 20}  (10 rows, 20 SLM phases each)
    W_k = E(phi_w[k])     complex weights for class k

Input encoding (selectable):
  patch-scan (default): raw 28x28 pixels split into 40 patches of 20 pixels each;
      E-field encoded and coherently accumulated -> x in C^20.
      Uses all 784 pixels -- no PCA compression.
  pca (--pca flag): PCA-20 projects 784 pixels to 20 principal components first.

Forward model:
    S_k = W_k . x         coherent sum
    detection mode:
      'intensity': I_k = |S_k|^2   square-law (physical, Rb 420 nm detector)
      'amplitude': I_k = |S_k|     linear amplitude (no squaring)
    y_k = I_k/(I_sat+I_k) saturable absorption in [0,1)
    pred = argmax_k y_k

Each detection mode is a separate Trackio run for side-by-side comparison.

Usage
-----
  python -m rbONN.train_sim                              # both modes, patch-scan
  python -m rbONN.train_sim --mode intensity             # single mode
  python -m rbONN.train_sim --epochs 300 --name "exp1"  # named experiment
  python -m rbONN.train_sim --pca                        # use PCA-20 instead
  python -m rbONN.train_sim --epochs 200 --hw-aware
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import trackio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import confusion_matrix, launch_trackio, print_model_summary
from .data import load_mnist
from .pca_encoder import PCAEncoder
from .twin import RbONNTwin, RbONNDeep, DETECTION_MODES

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
N_CLASSES = 10
TRACKIO_PROJECT = "rbONN_sim"
PATCH_SIZE = 20   # SLM spatial channels = pixels per patch


def _inject_noise(phi: torch.Tensor, sigma_rad: float, bits: int = 10) -> torch.Tensor:
    n = 2 ** bits
    phi_q = phi + (torch.round(phi / (2 * math.pi) * n) / n * (2 * math.pi) - phi).detach()
    return phi_q + torch.randn_like(phi_q) * sigma_rad


# ── Training -----------------------------------------------------------------

def train_one(
    detection_mode: str,
    A_tr: torch.Tensor,
    y_tr_t: torch.Tensor,
    A_te: torch.Tensor,
    y_te_t: torch.Tensor,
    device: torch.device,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-2,
    use_patches: bool = True,
    n_in: int = PATCH_SIZE,
    hw_aware: bool = False,
    phase_noise_rad: float = 0.05,
    save_path: Path | None = None,
    run_tag: str = "",
    arch: str = "single",
) -> dict:
    hw_suffix = "_hw_aware" if hw_aware else ""
    run_name = f"{run_tag}{hw_suffix}" if run_tag else f"{detection_mode}{hw_suffix}"

    if arch == "deep":
        encoding = "deep_4x4_patches_3layer_satsigmoid"
        arch_str = "RbONNDeep: 49x16 + 7x7 + 7, measured 420nm sigmoid (8402 params)"
    elif use_patches:
        encoding = f"patch_scan_{PATCH_SIZE}px_x{800 // PATCH_SIZE}_patches"
        arch_str = f"patch({PATCH_SIZE}px) x {N_CLASSES} weight rows"
    else:
        encoding = f"pca_{n_in}"
        arch_str = f"pca-{n_in} x {N_CLASSES} weight rows"

    trackio.init(
        project=TRACKIO_PROJECT,
        name=run_name,
        config={
            "arch": arch,
            "detection_mode": detection_mode,
            "encoding": encoding,
            "n_classes": N_CLASSES,
            "architecture": arch_str,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hw_aware": hw_aware,
        },
    )

    loader = DataLoader(
        TensorDataset(A_tr, y_tr_t),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    if arch == "deep":
        model = RbONNDeep(n_out=N_CLASSES, detection_mode=detection_mode).to(device)
    else:
        model = RbONNTwin(
            n_in=n_in,
            n_out=N_CLASSES,
            detection_mode=detection_mode,
            patch_size=PATCH_SIZE if use_patches else 0,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {arch_str}  |  {n_params} parameters")
    print_model_summary(model, (1, A_tr.shape[1]), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    history = []   # list of per-epoch dicts, written to JSON at end

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for a_batch, y_batch in loader:
            optimizer.zero_grad()
            if hw_aware:
                phi_noisy = _inject_noise(model.phi_w, sigma_rad=phase_noise_rad)
                logits = model(a_batch, phi_override=phi_noisy)
            else:
                logits = model(a_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(a_batch)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_te = model(A_te)
            pred = logits_te.argmax(dim=1)
            acc = (pred == y_te_t).float().mean().item()
            avg_loss = running_loss / len(A_tr)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        ep_log = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "test_acc": round(acc, 6),
            "best_acc": round(best_acc, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        i_sat = model.I_sat_value()
        if i_sat == i_sat:   # not nan
            ep_log["I_sat"] = round(i_sat, 6)

        # JSON history: only every 10th epoch (plus first and last) to keep it compact
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            history.append(ep_log)

        # Trackio gets every epoch for a smooth live curve
        trackio.log({**ep_log, "I_sat": i_sat})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            isat_str = f"  I_sat={i_sat:.3f}" if i_sat == i_sat else ""
            print(f"    [{run_name}] epoch {epoch+1:4d}/{epochs}  "
                  f"loss={avg_loss:.4f}  acc={acc:.2%}  best={best_acc:.2%}{isat_str}")

    # -- Restore best, final eval ---------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_te = model(A_te)
        pred = logits_te.argmax(dim=1)

    cm = confusion_matrix(pred, y_te_t, N_CLASSES)

    per_class_metrics = {}
    for k in range(N_CLASSES):
        per_class_metrics[f"class_{k}_acc"] = cm[k, k] / cm[k].sum()
    trackio.log({**per_class_metrics, "epoch": epochs})

    rows = "\n".join(
        f"| {k} | {cm[k,k]/cm[k].sum():.1%} | {cm[k].sum()} |"
        for k in range(N_CLASSES)
    )
    worst = min(range(N_CLASSES), key=lambda k: cm[k, k] / cm[k].sum())
    best_cls = max(range(N_CLASSES), key=lambda k: cm[k, k] / cm[k].sum())
    report = trackio.Markdown(f"""# Run complete -- {run_name}

| Metric | Value |
|--------|-------|
| Best test accuracy | {best_acc:.2%} |
| Detection mode | `{detection_mode}` |
| Encoding | {encoding} |
| Architecture | {PATCH_SIZE}px patches x {N_CLASSES} weight rows (10x{n_in} phases) |
| I_sat (learned) | {model.I_sat_value():.4f} |
| Hardware-aware | {hw_aware} |

## Per-class accuracy (best checkpoint)
| Class | Accuracy | Count |
|-------|----------|-------|
{rows}

Best class: **{best_cls}** ({cm[best_cls,best_cls]/cm[best_cls].sum():.1%})
Worst class: **{worst}** ({cm[worst,worst]/cm[worst].sum():.1%})
""")
    trackio.log({"summary": report, "epoch": epochs})
    trackio.finish()

    # Save a human-readable metrics JSON so training history is always inspectable
    metrics_path = OUTPUT_DIR / f"metrics_{run_name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "detection_mode": detection_mode,
            "encoding": encoding,
            "best_acc": best_acc,
            "I_sat": model.I_sat_value(),
            "hw_aware": hw_aware,
            "epochs": epochs,
            "per_class_acc": {str(k): float(cm[k, k] / cm[k].sum()) for k in range(N_CLASSES)},
            "history": history,
        }, f, indent=2)
    print(f"  Metrics -> {metrics_path}")

    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "encoding": encoding,
            "n_in": n_in,
            "patch_size": PATCH_SIZE if use_patches else 0,
            "n_out": N_CLASSES,
            "detection_mode": detection_mode,
        }, sp)
        print(f"  Saved model -> {sp}")

    return {"best_acc": best_acc, "model": model, "cm": cm}


# ── Entry point --------------------------------------------------------------

def train(
    modes: list[str] | None = None,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-2,
    use_patches: bool = True,
    n_pca: int = 20,
    hw_aware: bool = False,
    phase_noise_rad: float = 0.05,
    save_path: Path | None = None,
    name: str = "",
    arch: str = "single",
) -> dict[str, dict]:
    if modes is None:
        modes = list(DETECTION_MODES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Launch trackio dashboard in background so it's live during training
    launch_trackio(TRACKIO_PROJECT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else ""))

    # -- Data -----------------------------------------------------------------
    X_tr, y_tr, X_te, y_te = load_mnist()

    if arch == "deep" or use_patches:
        # Raw 28x28 pixels -- deep network handles patching internally
        A_tr = torch.from_numpy(X_tr).to(device)   # (60000, 784)
        A_te = torch.from_numpy(X_te).to(device)   # (10000, 784)
        n_in = PATCH_SIZE
        if arch == "deep":
            print("Encoding: raw 28x28 -> 4x4 spatial patches (handled by RbONNDeep)")
        else:
            n_patches = 800 // PATCH_SIZE
            print(f"Encoding: raw 28x28, raster patch-scan {PATCH_SIZE}px x {n_patches} patches")
    else:
        # PCA-20 encoding (legacy)
        encoder_path = OUTPUT_DIR / "pca_encoder.joblib"
        if encoder_path.exists():
            encoder = PCAEncoder.load(encoder_path)
            print(f"Loaded PCA encoder from {encoder_path}")
        else:
            print(f"Fitting PCA-{n_pca} on {len(X_tr)} training images ...")
            encoder = PCAEncoder(n_components=n_pca)
            encoder.fit(X_tr)
            encoder.save(encoder_path)
            var = encoder.explained_variance_ratio().sum()
            print(f"  explained variance: {var:.1%}")
        A_tr = torch.from_numpy(encoder.transform(X_tr)).to(device)
        A_te = torch.from_numpy(encoder.transform(X_te)).to(device)
        n_in = n_pca

    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    # -- Train each mode (separate Trackio run each) --------------------------
    histories: dict[str, dict] = {}
    for i, mode in enumerate(modes):
        print(f"\n=== Detection mode: {mode} ===")
        sp = save_path if i == 0 else None
        histories[mode] = train_one(
            mode, A_tr, y_tr_t, A_te, y_te_t, device,
            epochs=epochs, batch_size=batch_size, lr=lr,
            use_patches=use_patches, n_in=n_in,
            hw_aware=hw_aware, phase_noise_rad=phase_noise_rad,
            save_path=sp, run_tag=name, arch=arch,
        )
        print(f"  Best accuracy ({mode}): {histories[mode]['best_acc']:.2%}")

    return histories


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=list(DETECTION_MODES) + ["both"], default="intensity")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--name", type=str, default="",
                   help="Experiment name tag shown in trackio (e.g. 'patch_v1')")
    p.add_argument("--arch", choices=["single", "deep"], default="single",
                   help="'single': 1-layer RbONNTwin; 'deep': 3-layer RbONNDeep with 4x4 patches")
    p.add_argument("--pca", action="store_true",
                   help="Use PCA-20 encoding instead of raw patch scan (single arch only)")
    p.add_argument("--n-pca", type=int, default=20,
                   help="Number of PCA components (only used with --pca)")
    p.add_argument("--hw-aware", action="store_true")
    p.add_argument("--noise-rad", type=float, default=0.05)
    p.add_argument("--save", type=Path, default=OUTPUT_DIR / "twin.pt")
    args = p.parse_args()
    modes = list(DETECTION_MODES) if args.mode == "both" else [args.mode]
    train(
        modes=modes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_patches=not args.pca,
        n_pca=args.n_pca,
        hw_aware=args.hw_aware,
        phase_noise_rad=args.noise_rad,
        save_path=args.save,
        name=args.name,
        arch=args.arch,
    )


if __name__ == "__main__":
    main()
