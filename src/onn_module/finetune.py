"""MNIST CNN with a digital or an optical classifier head.

The convolutional trunk is identical in all three models; only the head that
turns 16x7x7 feature maps into 10 logits changes.

  bench    Flatten(784) -> Linear(784, 10)
           The ordinary digital baseline.  784*10 + 10 = 7,850 parameters.

  onn784   Flatten(784) -> ONN(784 -> 10)
           The same map, done optically.  The bench has 10 channels and can
           hold a 10x10 weight block at a time, so 784 inputs run as 79
           successive shots whose detector readings are summed.

  onn49    mean over the 16 channels -> Flatten(49) -> ONN(49 -> 10)
           Collapse the channel axis first, so the same map needs only 5
           shots.  Averaging over channels is exactly the big Linear with its
           weights tied across channels: 16x cheaper, and it throws away
           *which* kernel fired at a pixel while keeping *where*.

Two consequences of going optical, both deliberate:

  * No bias.  A logit is nothing but the weighted coherent quadratic sum
    Y = |sum_k eta_k sqrt(x_k w_k) e^{i(u_k+v_k+Phi_k)}|^2 summed over shots.
    The only digital step after the detector is a per-output gain (readout
    "scale"), because Y is of order eta^2 ~ 0.1 V and a softmax over a 0.1 V
    spread cannot train.  --onn-readout none removes even that.

  * No negative weights.  w lives in [0, 1], so Y >= 0 always.  Pass
    --onn-differential to run each shot twice and subtract (Y+ - Y-), which
    is how a signed weight is realised on the bench.

Every run is logged to trackio (project --project, default "rbonn-mnist"):
config, per-epoch loss / val accuracy / lr, the final test numbers, and the
plots.  Browse them with

    trackio show --project rbonn-mnist

and --no-trackio turns it off.  The same numbers also land in runs/<model>/ as
config.json, metrics.csv (written every epoch, so it survives a crash),
result.json, model.pt and PNGs, so nothing depends on the tracker being up.
--model all trains all three and writes runs/compare.json plus a chart.

Gaussian noise of std --noise is added to every input image, training and
test alike, after ToTensor and before clamping back to [0, 1].  The final
table reports accuracy on the clean test set and on the noisy one, so the
comparison says something about robustness and not just peak accuracy.

Runs in the rbonn-ml conda env (torch, torchvision, sklearn), not in rbonn.

    python src/onn_module/finetune.py --model all --epochs 15 --noise 0.1
    python src/onn_module/finetune.py --model onn49 --onn-differential
    python src/onn_module/finetune.py --model bench --eval

After a recalibration, carry a trained head over to the new eta/Phi instead of
starting again -- the checkpoint holds weights only, so --calib is what sets the
calibration and the run reports the eta it ends up with:

    python src/onn_module/finetune.py --model onn784 --tag recal --epochs 3
        --finetune runs/onn784_ctrl/model.pt
        --calib src/calib_data/<run>/calib_step7_result_<stamp>.json
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

if __package__ in (None, ""):      # run as a script: put src/ on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onn_module.layer import ONNLayer, onn_linear  # noqa: E402
from onn_module.params import ONNParams  # noqa: E402

try:
    import trackio
except ImportError:  # the runs still write config/metrics/result to disk
    trackio = None

# Data and outputs go under the repo root, not next to this file: src/ holds
# code, and a 60k-image MNIST download inside a package directory is not code.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = str(REPO_ROOT / "data")
RUNS_ROOT = str(REPO_ROOT / "runs")
LEGACY_WEIGHTS = str(REPO_ROOT / "runs" / "checkpoints" / "part2.pt")
#: The calibration a run defaults to.  Point --calib at the newest step-7 or
#: step-8 result; a step-8 file also carries the fitted Y_sat, which a step-7
#: one does not (see onn_module.params).
DEFAULT_CALIB = str(REPO_ROOT / "src" / "calib_data" / "calib_0917_1617"
                    / "calib_step7_result_0917_1831.json")

MODELS = ("bench", "onn784", "onn49")


# ------------------------------------------------------------------ data ----
class AddGaussianNoise:
    """Sensor noise on the input image, applied after ToTensor.

    Clamped back to [0, 1] because that is both what a pixel means and what
    the optical input arm can encode -- x_k outside [0, 1] has no phase.
    """

    def __init__(self, std):
        self.std = float(std)

    def __call__(self, t):
        if self.std <= 0:
            return t
        return (t + torch.randn_like(t) * self.std).clamp(0.0, 1.0)

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std})"


def make_transforms(noise=0.0):
    """(train, eval) transforms.  Augmentation is train-only; noise is on both.

    RandomAffine works on the PIL image, so it has to run before ToTensor.
    Validation and test must see un-augmented digits, otherwise the score
    measures the augmentation rather than the model -- but they *do* get the
    noise, since noise is part of the task we are measuring robustness to.
    """
    train = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,             # MNIST digits are upright; 10 deg is plenty
            translate=(0.1, 0.1),   # +/- ~3 pixels
            scale=(0.9, 1.1),
            shear=5,
        ),
        transforms.ToTensor(),
        AddGaussianNoise(noise),
    ])
    evaluate = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(noise)])
    return train, evaluate


def load_test_set(noise=0.0):
    """The official 10k test set.  Used once, at the end."""
    _, eval_tf = make_transforms(noise)
    return datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=eval_tf)


def load_train_val(noise=0.0, n_subset=None, val_frac=0.1):
    """Split MNIST's 60k training images into a train part and a validation part.

    Two dataset objects over the *same* files, because the two splits need
    different transforms: the training part gets augmented, the validation
    part must not be.  Each then gets a disjoint slice of one seeded
    permutation, so no image appears in both.
    """
    train_tf, eval_tf = make_transforms(noise)
    train_pool = datasets.MNIST(root=DATA_ROOT, train=True, download=True,
                                transform=train_tf)
    val_pool = datasets.MNIST(root=DATA_ROOT, train=True, download=True,
                              transform=eval_tf)

    # Fixed seed, so every model in a comparison sees the identical split.
    idx = torch.randperm(len(train_pool), generator=torch.Generator().manual_seed(0))
    if n_subset is not None and n_subset < len(idx):
        idx = idx[:n_subset]

    n_val = max(1, int(len(idx) * val_frac))
    return Subset(train_pool, idx[n_val:].tolist()), Subset(val_pool, idx[:n_val].tolist())


# ----------------------------------------------------------------- model ----
class ChannelMean(nn.Module):
    """(B, 16, 7, 7) -> (B, 1, 7, 7).  784 features become 49.

    Equivalent to the full Linear(784, 10) with its weights tied across the
    channel axis, so nothing is gained that the big layer could not already
    do -- the point is that 49 inputs is 5 optical shots instead of 79.
    """

    def forward(self, x):
        return x.mean(dim=1, keepdim=True)


def build_model(kind="bench", p_conv=0.15, p_fc=0.3, p_head=None,
                onn_params=None, **onn_kw):
    """Shared conv trunk (1x28x28 -> 16x7x7), then one of three heads.

        Conv2d(1, 8, 3, padding=1)   -> 8 x 28 x 28
        BatchNorm2d(8) / ReLU        -> same shape
        MaxPool2d(2)                 -> 8 x 14 x 14
        Conv2d(8, 16, 3, padding=1)  -> 16 x 14 x 14
        BatchNorm2d(16) / ReLU       -> same shape
        MaxPool2d(2)                 -> 16 x 7 x 7
        <head>                       -> 10 class scores

    The convs use bias=False: BatchNorm subtracts the mean straight after,
    which erases any bias, then adds its own learnable shift.  Dropout2d
    drops whole channels rather than pixels -- neighbouring pixels in a
    feature map are so correlated that dropping them singly removes almost
    no information.
    """
    if kind not in MODELS:
        raise ValueError(f"model: {kind!r}, expected one of {MODELS}")
    # Dropout in front of an ONN head is not the same operation as dropout in
    # front of a Linear one.  Inverted dropout divides the survivors by 1-p so
    # that the *linear* sum keeps its mean; the ONN output is |sum|^2, and the
    # same rescaling leaves it at ~0.79x its eval value.  The head then trains
    # under one gain and is evaluated under another, which is enough to tip it
    # into the all-channels-dark collapse.  So p_head defaults to 0.
    p_head = p_fc if p_head is None else p_head

    layers = [
        nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(p_conv),
        nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(p_conv),
    ]
    if kind == "bench":
        layers += [nn.Flatten(), nn.Dropout(p_fc), nn.Linear(16 * 7 * 7, 10)]
    elif kind == "onn784":
        layers += [nn.Flatten(), nn.Dropout(p_head),
                   onn_linear(16 * 7 * 7, 10, onn_params, **onn_kw)]
    else:  # onn49
        layers += [ChannelMean(), nn.Flatten(), nn.Dropout(p_head),
                   onn_linear(7 * 7, 10, onn_params, **onn_kw)]
    return nn.Sequential(*layers)


def lr_scale(args):
    """How much to soften the learning rates.  1.0 for a fresh run.

    A fine-tune restarts the cosine schedule at its peak, and the peak is sized
    for a randomly initialised head.  Applied to an already-trained one it is a
    large step in epoch 1 -- the same thing --grad-clip and --onn-clamp exist to
    survive -- and what it destroys is the model being fine-tuned.
    """
    return args.finetune_lr_scale if args.finetune else 1.0


def param_groups(model, args):
    """Split the parameters so an ONN head can run at its own learning rate."""
    scale = lr_scale(args)
    head = model[-1]
    if not isinstance(head, ONNLayer):
        return model.parameters()
    head_ids = {id(p) for p in head.parameters()}
    trunk = [p for p in model.parameters() if id(p) not in head_ids]
    return [dict(params=trunk, lr=args.lr * scale),
            dict(params=list(head.parameters()),
                 lr=(args.onn_lr or args.lr) * scale)]


@torch.no_grad()
def calibrate_readout(model, loader, device, target_std=1.0, floor=None):
    """Give an ONN head a detector gain that a softmax can actually use.

    Runs one batch through the frozen trunk, catches what reaches the head,
    and rescales the head's bias-free output gain so the spread across the 10
    logits is ~target_std.  Without this the ONN heads sit at chance forever:
    see ONNLayer.calibrate_readout.
    """
    head = model[-1]
    if not isinstance(head, ONNLayer) or head.readout == "none":
        return None
    caught = []
    handle = head.register_forward_pre_hook(lambda m, inp: caught.append(inp[0].detach()))
    was_training = model.training
    model.eval()  # no dropout, so the gain is set from the features it will see
    images, _ = next(iter(loader))
    model(images.to(device))
    handle.remove()
    model.train(was_training)
    spread = head.calibrate_readout(caught[0], target_std, only_if_below=floor)
    if floor is None:
        print(f"  readout gain: logit spread was {spread:.4f} V -> scaled to {target_std}")
    elif spread < floor:
        print(f"    AGC: logit spread had fallen to {spread:.3f} -> rescaled to "
              f"{target_std} (the head was going dark)")
    return spread


def load_checkpoint(model, path, device, kind, missing=None):
    """Load weights into `model`, reporting what the file did and did not carry.

    A checkpoint holds weights only.  The calibration is registered
    non-persistent (see ONNLayer.__init__), so eta and Phi always come from
    --calib and never from the file -- which is what makes fine-tuning against a
    fresh step 6/7 mean anything.  This prints the eta actually in force, because
    "did my recalibration take effect" is the first question a fine-tune raises
    and the answer used to be silently no.
    """
    if not os.path.exists(path):
        raise SystemExit(missing or f"--finetune {path} not found")
    state = torch.load(path, map_location=device, weights_only=True)
    stale = [k for k in state if k.endswith((".eta", ".phi"))]
    for k in stale:
        state.pop(k)
    try:
        model.load_state_dict(state)
    except RuntimeError as err:
        raise SystemExit(f"{path} does not match this file's {kind} model -- "
                         f"retrain it\n\n{err}")
    print(f"  loaded weights from {path}")
    if stale:
        # Written by a version that saved the calibration as a persistent
        # buffer.  Dropping those keys is the correction, not a workaround:
        # keeping them would restore the calibration the run was trained under.
        print(f"    ignored {len(stale)} calibration entr"
              f"{'y' if len(stale) == 1 else 'ies'} in the file ({', '.join(stale)})"
              f" -- eta/Phi come from --calib")
    head = model[-1]
    if isinstance(head, ONNLayer):
        print(f"    calibration in force: eta [{head.eta.min():.4f}, "
              f"{head.eta.max():.4f}], y_sat {head.y_sat}")
    return model


def get_device():
    """NVIDIA GPU, then Apple-silicon GPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------------------------------------- eval / io ----
def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()  # BatchNorm switches to running stats, Dropout turns off
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def collect_predictions(model, loader):
    device = next(model.parameters()).device
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds.append(model(images.to(device)).argmax(dim=1).cpu())
            trues.append(labels)
    return torch.cat(preds), torch.cat(trues)  # back on CPU for sklearn


def run_dir(kind, tag=""):
    d = os.path.join(RUNS_ROOT, kind + (f"_{tag}" if tag else ""))
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def write_metrics(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "lr", "train_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)


def save_fig(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight")


# ----------------------------------------------------------------- plots ----
def plot_sample_predictions(model, test_ds, n=10):
    """n test digits with the predicted label (red = wrong)."""
    device = next(model.parameters()).device
    model.eval()
    fig, axes = plt.subplots(2, n // 2, figsize=(1.4 * (n // 2), 3.2))
    for ax, (image, true) in zip(axes.ravel(), test_ds):
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"pred {pred}", color="green" if pred == true else "red")
        ax.axis("off")
    fig.suptitle("Test digits and the prediction (red = wrong)")
    fig.tight_layout()
    return fig


def plot_augmented_samples(train_ds, n=10):
    """What augmentation and noise actually do to the training digits."""
    fig, axes = plt.subplots(2, n // 2, figsize=(1.4 * (n // 2), 3.2))
    for ax, (image, label) in zip(axes.ravel(), train_ds):
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"label {label}")
        ax.axis("off")
    fig.suptitle("Training digits after augmentation + noise")
    fig.tight_layout()
    return fig


def plot_training_curves(losses, val_accs, title=""):
    """Train loss and validation accuracy on a shared epoch axis.

    Loss still falling while validation accuracy flattens or drops means the
    model has started memorising rather than learning.
    """
    fig, ax_loss = plt.subplots()
    epochs = range(1, len(losses) + 1)
    ax_loss.plot(epochs, losses, marker="o", color="tab:green", label="train loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy", color="tab:green")
    ax_loss.tick_params(axis="y", labelcolor="tab:green")

    ax_acc = ax_loss.twinx()
    ax_acc.plot(epochs, [a * 100 for a in val_accs], marker="s", color="tab:blue")
    ax_acc.set_ylabel("validation accuracy (%)", color="tab:blue")
    ax_acc.tick_params(axis="y", labelcolor="tab:blue")
    ax_loss.set_title(title)
    return fig


def plot_confusion(preds, trues, title=""):
    # Imported here rather than at the top: sklearn is needed for this one plot
    # and nothing else, so its absence should cost a figure, not the module.
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(trues.numpy(), preds.numpy(), labels=range(10))
    disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
    disp.plot(cmap="Blues", colorbar=False)
    disp.ax_.set_title(title or "Confusion matrix (rows = true, cols = predicted)")
    return disp.figure_


def plot_comparison(results, noise):
    """Validation curves and the final accuracies, all three models together."""
    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(11, 4))

    for r in results:
        if r.get("val_accs"):
            ax_curve.plot(range(1, len(r["val_accs"]) + 1),
                          [a * 100 for a in r["val_accs"]], marker="o", label=r["model"])
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("validation accuracy (%)")
    ax_curve.set_title("Validation accuracy")
    ax_curve.legend()
    ax_curve.grid(alpha=0.3)

    names = [r["model"] for r in results]
    pos = range(len(names))
    ax_bar.bar([p - 0.2 for p in pos], [100 * r["test_acc"] for r in results],
               width=0.4, label="clean test")
    ax_bar.bar([p + 0.2 for p in pos], [100 * r["test_acc_noisy"] for r in results],
               width=0.4, label=f"noisy test (std {noise})")
    ax_bar.set_xticks(list(pos))
    ax_bar.set_xticklabels(names)
    ax_bar.set_ylabel("test accuracy (%)")
    ax_bar.set_ylim(min(90, min(100 * r["test_acc_noisy"] for r in results) - 5), 100)
    ax_bar.set_title("Final accuracy")
    ax_bar.legend()
    fig.tight_layout()
    return fig


# -------------------------------------------------------------- training ----
def train_one(kind, args, device, onn_params, data):
    """Train one model, write everything it produced to its run directory."""
    train_ds, val_ds, test_ds, test_noisy_ds = data
    out = run_dir(kind, args.tag)
    ckpt = os.path.join(out, "model.pt")

    # Both generators: torch.manual_seed does NOT reseed the MPS device RNG,
    # and that is the one Dropout draws from on a Mac.  Without this line a
    # run's dropout stream depends on whatever ran before it in the same
    # process, so --model all and --model onn784 are different experiments and
    # --seed does not actually vary anything.
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    model = build_model(
        kind,
        p_conv=args.dropout / 2,
        p_fc=args.dropout,
        p_head=args.onn_dropout,
        onn_params=onn_params,
        input_map="tanh",          # post-ReLU features: 0 stays off
        input_gain=True,           # learnable drive level per channel
        differential=args.onn_differential,
        readout=args.onn_readout,
        drive_bits=args.drive_bits,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    head = model[-1]
    print(f"\n=== {kind} ===")
    print(f"  {n_params:,} parameters   head: {head}")

    config = dict(vars(args))
    config.update(model=kind, n_params=n_params, device=str(device),
                  head=str(head), run_dir=out)
    if isinstance(head, ONNLayer):
        config.update(n_tiles=head.n_tiles, n_channels=head.in_features,
                      # One detector: the O weight vectors share the K
                      # channels and are measured one after another, so a
                      # sample costs n_tiles * out_features readings, not
                      # n_tiles.
                      shots_per_sample=head.n_tiles * head.out_features
                                       * (2 if head.differential else 1))
    write_json(os.path.join(out, "config.json"), config)

    track = trackio is not None and not args.no_trackio
    if track:
        trackio.init(project=args.project, name=kind + (f"-{args.tag}" if args.tag else ""),
                     config=config)

    test_loader = DataLoader(test_ds, batch_size=256)
    noisy_loader = DataLoader(test_noisy_ds, batch_size=256)

    losses, val_accs, rows = [], [], []
    best_val, best_epoch, seconds = -1.0, 0, 0.0

    if args.eval:
        load_checkpoint(model, ckpt, device, kind,
                        missing=f"{ckpt} not found -- train {kind} once without --eval")
        print("  (skipping training)")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)
        if args.finetune:
            if os.path.abspath(args.finetune) == os.path.abspath(ckpt):
                raise SystemExit(
                    f"--finetune {args.finetune} is this run's own checkpoint, so "
                    f"training would overwrite the file it started from.\n"
                    f"Pass a different --tag (the run directory is {out}).")
            load_checkpoint(model, args.finetune, device, kind)
        # AFTER any load, and that ordering is the point: a recalibration moves
        # eta, Y scales as eta^2, and the out_scale saved in the checkpoint was
        # fitted to the old one.  Re-running the one-shot gain calibration is
        # exactly what a fine-tune against a new step 6/7 needs.
        calibrate_readout(model, train_loader, device, args.onn_logit_std)
        loss_fn = nn.CrossEntropyLoss()
        # AdamW, not Adam: it applies weight decay as true decay rather than
        # folding it into the gradient, which is what "decay" should mean.
        #
        # The ONN head gets its own, larger learning rate.  Its weights are
        # SLM phases, and w = sin^2((pi/2) sigmoid(theta)) compresses the
        # gradient by dw/dtheta <= 0.2, so at the trunk's rate the head
        # converges several times slower than the conv stack it sits on.
        # The penalty needs the head's *input*, which model(images) does not
        # hand back.  A pre-hook stashes the live tensor, so the drive is
        # recomputed on the same graph and the gradient reaches in_scale.
        drive_on = isinstance(head, ONNLayer)
        captured = {}
        if drive_on:
            head.register_forward_pre_hook(
                lambda _m, inp: captured.__setitem__("x", inp[0]))

        optimizer = torch.optim.AdamW(param_groups(model, args),
                                      lr=args.lr * lr_scale(args),
                                      weight_decay=args.weight_decay)
        # Cosine: lr eases to ~0 so the last epochs settle instead of bouncing.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=args.epochs)
        t0 = time.time()
        for epoch in range(args.epochs):
            spread = None
            if args.onn_agc > 0:
                # One-sided gain control, once per epoch.  Cross-entropy can
                # descend into "predict nothing": every channel dark, every
                # logit equal, loss pinned at ln(10) -- which is *worse* than
                # where it came from, so it is an overshoot that it then
                # cannot undo.  Raising the gain when the spread collapses
                # gives the phases a gradient again.
                spread = calibrate_readout(model, train_loader, device,
                                           args.onn_logit_std,
                                           floor=args.onn_agc * args.onn_logit_std)
            model.train()  # BatchNorm updates running stats, Dropout is active
            epoch_loss, epoch_head = 0.0, 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                loss = loss_fn(model(images), labels)
                if drive_on and args.onn_drive > 0:
                    headroom = head.drive_headroom(captured["x"], args.onn_drive_top)
                    loss = loss + args.onn_drive * headroom
                    epoch_head += headroom.item()
                elif drive_on:      # diagnostic only, off the graph
                    with torch.no_grad():
                        epoch_head += head.drive_headroom(captured["x"],
                                                          args.onn_drive_top).item()
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if isinstance(model[-1], ONNLayer):
                    model[-1].clamp_(args.onn_clamp)  # keep gradients alive
                epoch_loss += loss.item()

            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()  # once per epoch, not per batch

            headroom = epoch_head / len(train_loader) if drive_on else None
            losses.append(epoch_loss / len(train_loader))
            val_accs.append(evaluate(model, val_loader))
            rows.append(dict(epoch=epoch + 1, lr=lr_now,
                             train_loss=losses[-1], val_acc=val_accs[-1]))
            write_metrics(os.path.join(out, "metrics.csv"), rows)  # survives a crash
            if track:
                trackio.log(dict(train_loss=losses[-1], val_acc=val_accs[-1],
                                 lr=lr_now, logit_spread=spread or 0.0,
                                 drive_headroom=headroom or 0.0),
                            step=epoch + 1)

            flag = ""
            if val_accs[-1] > best_val:
                # Keep the best epoch by validation accuracy, not merely the
                # last -- the final epoch is not always the strongest model.
                best_val, best_epoch = val_accs[-1], epoch + 1
                torch.save(model.state_dict(), ckpt)
                flag = "  <- best so far, saved"
            drive_txt = f"  drive {1 - headroom ** 0.5:.2f}" if drive_on else ""
            print(f"  epoch {epoch + 1:2d}/{args.epochs}  loss {losses[-1]:.4f}  "
                  f"val acc {val_accs[-1]:.2%}  lr {lr_now:.2e}{drive_txt}{flag}")
        seconds = time.time() - t0

        # Reload the best checkpoint so the test number describes what we kept.
        model.load_state_dict(torch.load(ckpt, map_location=device))
        if kind == "bench":  # part4 reloads this exact path
            os.makedirs(os.path.dirname(LEGACY_WEIGHTS), exist_ok=True)
            torch.save(model.state_dict(), LEGACY_WEIGHTS)

    acc = evaluate(model, test_loader)
    acc_noisy = evaluate(model, noisy_loader)
    print(f"  test acc {acc:.2%} (clean)   {acc_noisy:.2%} (noise std {args.noise})")

    result = dict(model=kind, n_params=n_params, best_val=best_val,
                  best_epoch=best_epoch, test_acc=acc, test_acc_noisy=acc_noisy,
                  train_seconds=round(seconds, 1), losses=losses, val_accs=val_accs,
                  run_dir=out, checkpoint=ckpt, config=config)
    write_json(os.path.join(out, "result.json"), result)

    title = f"{kind}  (test {acc:.1%}, {n_params:,} params)"
    if losses:
        save_fig(plot_training_curves(losses, val_accs, title),
                 os.path.join(out, "curves.png"))
    if train_ds is not None:
        save_fig(plot_augmented_samples(train_ds), os.path.join(out, "inputs.png"))
    save_fig(plot_sample_predictions(model, test_ds), os.path.join(out, "predictions.png"))
    preds, trues = collect_predictions(model, test_loader)
    save_fig(plot_confusion(preds, trues, title), os.path.join(out, "confusion.png"))

    if track:
        trackio.log(dict(test_acc=acc, test_acc_noisy=acc_noisy, best_val=best_val,
                         best_epoch=best_epoch, train_seconds=seconds))
        for png in ("curves", "inputs", "predictions", "confusion"):
            path = os.path.join(out, f"{png}.png")
            if os.path.exists(path):
                trackio.log({png: trackio.Image(path, caption=f"{kind} {png}")})
        trackio.finish()
    print(f"  wrote {out}")
    return result


def print_table(results, noise):
    head = f"{'model':<9}{'params':>9}{'best val':>11}{'test':>9}" \
           f"{f'test@{noise}':>12}{'sec':>8}"
    print("\n" + head)
    print("-" * len(head))
    for r in results:
        best = f"{r['best_val']:.2%}" if r["best_val"] >= 0 else "-"
        print(f"{r['model']:<9}{r['n_params']:>9,}{best:>11}"
              f"{r['test_acc']:>9.2%}{r['test_acc_noisy']:>12.2%}"
              f"{r['train_seconds']:>8.0f}")


# ------------------------------------------------------------------ main ----
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", choices=MODELS + ("all",),
                   help="which head to train (default: all three)")
    p.add_argument("--eval", action="store_true",
                   help="skip training: load runs/<model>/model.pt and report")
    p.add_argument("--tag", default="", help="suffix for the run directory")
    p.add_argument("--show", action="store_true", help="also open the plots")
    p.add_argument("--project", default="rbonn-mnist", help="trackio project name")
    p.add_argument("--no-trackio", action="store_true",
                   help="do not log to trackio (runs/ is written either way)")

    p.add_argument("--noise", type=float, default=0.1,
                   help="std of Gaussian noise added to every input image, "
                        "train and test (default: 0.1)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--n-train", type=int, default=None,
                   help="use only this many of the 60k training images")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3,
                   help="peak learning rate; BatchNorm lets us start high")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3,
                   help="dropout before the head; conv blocks use half")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--finetune", default=None, metavar="CHECKPOINT",
                   help="start from this model.pt instead of from scratch, then "
                        "train --epochs more.  For carrying a trained head onto a "
                        "fresh calibration: the file holds weights only, so "
                        "--calib decides eta/Phi and the detector gain is "
                        "re-fitted for the new eta.  Refuses to run if it would "
                        "overwrite the checkpoint it started from -- use --tag")
    p.add_argument("--finetune-lr-scale", type=float, default=0.1,
                   help="multiply both learning rates by this while fine-tuning "
                        "(default: 0.1).  The schedule restarts at its peak, and "
                        "a peak sized for a random head is a destructive first "
                        "step for a trained one")
    p.add_argument("--calib", default=DEFAULT_CALIB,
                   help="ONN calibration JSON: a step-7 result (eta, Phi) or a "
                        "step-8 / system-accuracy result, which also carries the "
                        "fitted Y_sat")
    p.add_argument("--y-sat", type=float, default=None,
                   help="atom saturation level in detector volts, overriding the "
                        "calibration file.  Default: whatever --calib says (a "
                        "step-8 result carries a fitted y_sat_v; a step-7 one "
                        "does not, and the layer then runs unsaturated).  At the "
                        "drive a trained onn784 actually uses, peak compression "
                        "is ~2%, so this is not a sensitive knob -- but a "
                        "hardcoded value that disagrees with the run it is quoted "
                        "against is worse than none")
    p.add_argument("--drive-bits", type=int, default=None,
                   help="quantize both SLM arms to 2**BITS settings per channel, "
                        "which is what the bench can actually display (the 10-pair "
                        "layout runs 4-bit).  Default None leaves the drive "
                        "continuous, i.e. optimistic: see ONNLayer._quantize for "
                        "what the uniform grid does and does not model")
    p.add_argument("--onn-readout", default="scale", choices=("none", "scale", "affine"),
                   help="digital post-detector processing; 'scale' is a gain "
                        "with no bias (default), 'none' is the raw Y -- which "
                        "cannot train, because a softmax over a 0.03 V spread "
                        "is flat.  Kept so you can see that for yourself")
    p.add_argument("--onn-lr", type=float, default=1e-2,
                   help="learning rate for the ONN head only; its SLM-phase "
                        "parametrisation needs ~5x the trunk's (default: 1e-2, "
                        "vs --lr 2e-3).  Sensitive: 2e-2 destabilises the "
                        "79-tile head in epoch 1 and it never recovers")
    p.add_argument("--onn-dropout", type=float, default=0.0,
                   help="dropout in front of an ONN head (default: 0).  Not the "
                        "same knob as --dropout: inverted dropout's 1/(1-p) "
                        "correction is derived for a linear layer and is wrong "
                        "for a quadratic one, so this biases train vs eval")
    p.add_argument("--onn-agc", type=float, default=0.25,
                   help="per-epoch automatic gain control: if the logit spread "
                        "falls below this fraction of --onn-logit-std, rescale "
                        "the detector gain back up (0 disables)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="clip the global gradient norm (0 disables).  The ONN "
                        "head sits behind a detector gain of ~70, so one large "
                        "step can drive every channel dark; from there every "
                        "logit is equal and the run is stuck at loss ln(10) "
                        "even though that is *worse* than where it came from")
    p.add_argument("--onn-clamp", type=float, default=4.0,
                   help="bound on the raw SLM phase variable, applied after "
                        "every step; without it the head can saturate every "
                        "channel dark and strand itself at loss = ln(10)")
    p.add_argument("--onn-logit-std", type=float, default=1.0,
                   help="target spread across the 10 logits after the one-shot "
                        "detector-gain calibration (default: 1.0)")
    p.add_argument("--onn-drive", type=float, default=0.0,
                   help="weight on the drive-occupancy reward.  LEAVE AT 0: "
                        "measured, it only costs accuracy.  A healthy onn784 "
                        "already drives 9 of 10 channels to 0.97-1.00 at the "
                        "top of their range, so there is nothing to win, and "
                        "at 0.1 the penalty gradient (~0.9) swamps cross-"
                        "entropy's (~0.004) and drags the conv trunk with it: "
                        "97.7% -> 61.7% one-sided, -> 15.7% two-sided.  The "
                        "headroom is logged as a diagnostic either way")
    p.add_argument("--onn-drive-top", type=float, default=0.01,
                   help="fraction of the batch counted as a channel's bright "
                        "tail for --onn-drive (default: top 1%%)")
    p.add_argument("--onn-differential", action="store_true",
                   help="two shots per tile, Y+ - Y-, so weights can be signed")
    return p.parse_args()


def main():
    args = parse_args()
    kinds = MODELS if args.model == "all" else (args.model,)

    device = get_device()
    print(f"device: {device}   noise std: {args.noise}")

    onn_params = None
    if any(k.startswith("onn") for k in kinds):
        if os.path.exists(args.calib):
            onn_params = ONNParams.from_json(args.calib, y_sat=args.y_sat)
            print(onn_params.summary())
        else:
            onn_params = ONNParams.ideal(10, y_sat=args.y_sat)
            print(f"WARNING: {args.calib} not found -- using an ideal 10-channel "
                  f"device (eta=1, Phi=0), so the ONN results are optimistic")

    # Clean and noisy copies of the test set; the clean one is the headline
    # number, the noisy one says how much the head leans on a quiet input.
    test_ds = load_test_set(noise=0.0)
    test_noisy_ds = load_test_set(noise=args.noise)

    train_ds = val_ds = None
    if not args.eval:
        train_ds, val_ds = load_train_val(noise=args.noise, n_subset=args.n_train,
                                          val_frac=args.val_frac)
        print(f"train {len(train_ds):,}   val {len(val_ds):,}   test {len(test_ds):,}")

    data = (train_ds, val_ds, test_ds, test_noisy_ds)
    results = [train_one(k, args, device, onn_params, data) for k in kinds]

    print_table(results, args.noise)
    if len(results) > 1:
        os.makedirs(RUNS_ROOT, exist_ok=True)
        write_json(os.path.join(RUNS_ROOT, f"compare{('_' + args.tag) if args.tag else ''}.json"),
                   dict(noise=args.noise, epochs=args.epochs, device=str(device),
                        results=results))
        save_fig(plot_comparison(results, args.noise),
                 os.path.join(RUNS_ROOT, "compare.png"))
        print(f"\nwrote {RUNS_ROOT}/compare.json and compare.png")

    if trackio is not None and not args.no_trackio:
        print(f"\ntrackio: browse these runs with  "
              f"trackio show --project {args.project}")
    elif trackio is None:
        print("\ntrackio not installed -- metrics are in runs/ only "
              "(pip install trackio)")

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
