"""Generate paper figures.

    ./venv/bin/python paper/figures.py

Writes into paper/: fig1_workflow.png (diagrams/graphviz), fig2_dataset.png (matplotlib),
fig3_architecture.png (matplotlib). Figure 1 needs graphviz on PATH (`brew install graphviz`).
"""
from pathlib import Path

OUT = Path(__file__).parent


def fig1_workflow():
    from diagrams import Diagram, Cluster, Edge
    from diagrams.programming.flowchart import (
        Database, PredefinedProcess, Document, Decision, InputOutput)

    graph_attr = {"fontsize": "16", "bgcolor": "white", "pad": "0.4", "splines": "spline"}
    with Diagram("", filename=str(OUT / "fig1_workflow"), show=False,
                 direction="LR", graph_attr=graph_attr, outformat="png"):
        with Cluster("Source corpora"):
            sources = [Database("RAS-Compound\n7,830 img / 119 cls"),
                       Database("Ekush\n367,018 img / 122 cls"),
                       Database("MatriVasha\n306,461 img / 115 cls")]

        with Cluster("Label reconciliation"):
            mapping = PredefinedProcess("folder ID → grapheme\n(per-source CSV)")
            nfc = PredefinedProcess("Unicode NFC\nnormalisation")
            union = Document("Unified manifest\n681,309 img / 256 cls")

        with Cluster("Preprocessing"):
            split = Decision("Stratified split\n80 / 10 / 10, seed 42")
            polarity = PredefinedProcess("Polarity\nnormalisation")
            augment = PredefinedProcess("Resize 224²\nrotate / affine / jitter")

        with Cluster("Architectures (identical protocol)"):
            models = [PredefinedProcess("Scratch CNN\n0.06M"),
                      PredefinedProcess("ResNet18\n11.3M"),
                      PredefinedProcess("ExtendedViT\n23.9M"),
                      PredefinedProcess("ViT-B/16\n86.3M")]

        evaluation = InputOutput("Shared test set\nacc · CER · macro-F1\nper-source · CPU latency")

        for s in sources:
            s >> Edge(color="gray40") >> mapping
        mapping >> nfc >> union >> split >> polarity >> augment
        for m in models:
            augment >> Edge(color="gray40") >> m
            m >> Edge(color="gray40") >> evaluation


def fig2_dataset():
    """(a) per-class frequency, log scale, with the rare-class cutoff marked; (b) source composition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter
    import sys
    sys.path.insert(0, str(OUT.parent))
    from train_merged import build_manifest

    samples, classes = build_manifest()
    counts = sorted(Counter(l for _, l in samples).values(), reverse=True)
    per_source = Counter("RAS-Compound" if "RAS-Comp" in p else
                         "Ekush" if "ekush" in p else "MatriVasha" for p, _ in samples)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.fill_between(range(len(counts)), counts, color="#4C72B0", alpha=.35)
    ax1.plot(counts, color="#2b4c7e", lw=1.4)
    ax1.axhline(100, color="#c44e52", ls="--", lw=1)
    ax1.text(len(counts) * .45, 118, "27 classes below 100 samples",
             color="#c44e52", fontsize=9)
    ax1.set_yscale("log")
    ax1.set_xlabel("Class rank")
    ax1.set_ylabel("Images (log scale)")
    ax1.set_title("(a) Per-class frequency", fontsize=11, loc="left")
    ax1.margins(x=0.01)

    names = ["Ekush", "MatriVasha", "RAS-Compound"]
    vals = [per_source[n] for n in names]
    bars = ax2.barh(names, vals, color=["#4C72B0", "#55A868", "#C44E52"], height=.55)
    for b, v in zip(bars, vals):
        ax2.text(v + 8000, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=9)
    ax2.set_xlabel("Images")
    ax2.set_xlim(0, max(vals) * 1.25)
    ax2.set_title("(b) Source composition", fontsize=11, loc="left")
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_dataset.png", dpi=200)
    plt.close(fig)


def fig3_architecture():
    """ExtendedViT data flow with tensor shapes — drawn as boxes so shapes stay legible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    stages = [
        ("Input\nimage", "3 × 224 × 224", "#E8E8E8"),
        ("ResNet18\nconv stages", "512 × 7 × 7", "#4C72B0"),
        ("Flatten\nto tokens", "49 × 512", "#6B8EBF"),
        ("+ CLS token\n+ pos. embed", "50 × 512", "#8AA9CE"),
        ("Transformer\nencoder × 4", "50 × 512", "#55A868"),
        ("CLS token\n→ LN → linear", "256 logits", "#C44E52"),
    ]

    fig, ax = plt.subplots(figsize=(12, 2.9))
    x, w, gap = 0.0, 1.7, 0.55
    for i, (label, shape, color) in enumerate(stages):
        dark = i in (1, 4, 5)
        ax.add_patch(FancyBboxPatch((x, 0.42), w, 0.86, boxstyle="round,pad=0.03,rounding_size=0.06",
                                    fc=color, ec="none"))
        ax.text(x + w / 2, 0.85, label, ha="center", va="center", fontsize=9.5,
                color="white" if dark else "#333", fontweight="bold" if dark else "normal")
        ax.text(x + w / 2, 0.28, shape, ha="center", va="center", fontsize=8.5,
                color="#555", family="monospace")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.06, 0.85), (x + w + gap - 0.06, 0.85),
                                         arrowstyle="-|>", mutation_scale=13, color="#888", lw=1.2))
        x += w + gap

    ax.text(x - gap - w / 2, 1.46, "self-attention over the whole glyph", ha="center",
            fontsize=8.5, color="#55A868", style="italic")
    ax.text(1.7 + gap + 1.7 / 2, 1.46, "ImageNet inductive bias", ha="center",
            fontsize=8.5, color="#4C72B0", style="italic")
    ax.set_xlim(-0.15, x - gap + 0.15)
    ax.set_ylim(0.1, 1.7)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _manifest():
    import sys
    sys.path.insert(0, str(OUT.parent))
    from train_merged import build_manifest
    return build_manifest()


def _source(path):
    return ("RAS-Compound" if "RAS-Comp" in path else
            "Ekush" if "ekush" in path else "MatriVasha")


def fig4_samples():
    """The same characters as collected by each source — the variation preprocessing must absorb."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from collections import defaultdict

    samples, classes = _manifest()
    by_class_source = defaultdict(lambda: defaultdict(list))
    for p, lab in samples:
        by_class_source[lab][_source(p)].append(p)

    SOURCES = ["RAS-Compound", "Ekush", "MatriVasha"]
    shared = [c for c, d in by_class_source.items() if len(d) == 3][:5]
    if not shared:  # fall back to whatever each source has
        shared = list(by_class_source)[:5]

    def best_sample(paths, n=12):
        """Most ink of the first n candidates — the raw first pick is sometimes a near-blank crop."""
        import numpy as np
        scored = []
        for p in paths[:n]:
            a = np.asarray(Image.open(p).convert("L"), dtype=float)
            ink = (a > 127).mean() if a.mean() < 127 else (a < 127).mean()
            scored.append((ink, p))
        return max(scored)[1] if scored else None

    fig, axes = plt.subplots(len(SOURCES), len(shared),
                             figsize=(1.5 * len(shared), 1.65 * len(SOURCES)))
    for r, src in enumerate(SOURCES):
        for c, cls in enumerate(shared):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            paths = by_class_source[cls].get(src, [])
            pick = best_sample(paths)
            if pick:
                ax.imshow(Image.open(pick).convert("L"), cmap="gray")
            else:
                ax.text(.5, .5, "—", ha="center", va="center", transform=ax.transAxes, color="#bbb")
            if r == 0:
                ax.set_title(classes[cls], fontsize=17, pad=7,
                             fontfamily=["Kohinoor Bangla", "Bangla MN", "sans-serif"])
            if c == 0:
                ax.set_ylabel(src, fontsize=9)
    fig.suptitle("Same character, three corpora — note the inverted ink polarity",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_samples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig5_preprocessing():
    """Raw glyph through polarity normalisation and three augmentation draws."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    import sys
    sys.path.insert(0, str(OUT.parent))
    from train_merged import PolarityNormalize, build_transforms

    samples, _ = _manifest()
    # one glyph per source so the polarity fix is visible on a light-on-dark and a dark-on-light case
    picks, seen = [], set()
    for p, _lab in samples:
        s = _source(p)
        if s not in seen:
            seen.add(s), picks.append((s, p))
        if len(seen) == 3:
            break

    train_tf, _ = build_transforms()
    fig, axes = plt.subplots(len(picks), 5, figsize=(9.5, 2.1 * len(picks)))
    titles = ["Raw input", "Polarity normalised", "Augmented (1)", "Augmented (2)", "Augmented (3)"]

    for r, (src, path) in enumerate(picks):
        raw = Image.open(path)
        axes[r, 0].imshow(raw.convert("L"), cmap="gray")
        axes[r, 1].imshow(PolarityNormalize()(raw), cmap="gray")
        torch.manual_seed(r)
        for c in range(2, 5):
            t = train_tf(raw)                       # denormalise for display
            axes[r, c].imshow((t * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).numpy())
        for c in range(5):
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9.5)
        axes[r, 0].set_ylabel(src, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_preprocessing.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _parse_log(path):
    import re
    if not path.exists():
        return []
    pat = re.compile(r"Epoch \[(\d+)/\d+\] Train Loss: ([\d.]+) Acc: ([\d.]+) \| "
                     r"Val Loss: ([\d.]+) Acc: ([\d.]+)")
    return [tuple(map(float, m)) for m in pat.findall(path.read_text(errors="replace"))]


LOGS = OUT.parent / "logs"
METHODS = [("cnn", "Scratch CNN", "#C44E52"), ("resnet18", "ResNet18", "#4C72B0"),
           ("extended_vit", "ExtendedViT (ours)", "#55A868"), ("vit", "ViT-B/16", "#8172B2")]


def fig6_training_curves():
    """Validation accuracy and loss per epoch at the equal-budget setting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    # the three pretrained models sit within 2 points of each other, invisible on a 0-1 axis
    inset = ax1.inset_axes([0.42, 0.30, 0.55, 0.42])
    partial = []
    for key, label, color in METHODS:
        rows = _parse_log(LOGS / f"{key}.log")
        if not rows:
            continue
        ep = [r[0] for r in rows]
        ax1.plot(ep, [r[4] for r in rows], "-o", color=color, ms=3.5, lw=1.6, label=label)
        ax2.plot(ep, [r[3] for r in rows], "-o", color=color, ms=3.5, lw=1.6, label=label)
        if key != "cnn":
            inset.plot(ep, [r[4] for r in rows], "-o", color=color, ms=2.8, lw=1.4)
        if len(rows) < 8:
            partial.append(f"{label}: {len(rows)}/8 epochs")

    inset.set_ylim(0.89, 0.99)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=.25, lw=.5)
    inset.set_title("pretrained models, zoomed", fontsize=7.5, pad=3)

    ax1.set_xlabel("Epoch"), ax1.set_ylabel("Validation accuracy")
    ax1.set_title("(a) Validation accuracy", fontsize=11, loc="left")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8.5, loc="upper left", frameon=False, bbox_to_anchor=(0, .92))
    ax2.set_xlabel("Epoch"), ax2.set_ylabel("Validation loss")
    ax2.set_title("(b) Validation loss", fontsize=11, loc="left")
    ax2.legend(fontsize=8.5, frameon=False)
    for ax in (ax1, ax2):
        ax.grid(alpha=.25, lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if partial:
        fig.text(.5, -.02, "Runs still in progress — " + "; ".join(partial),
                 ha="center", fontsize=8, color="#c44e52")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig7_results():
    """(a) accuracy by method and budget; (b) the data-efficiency result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Full-corpus numbers are best-val from the July runs; capped come from this chain's logs.
    FULL = {"resnet18": 0.9768, "extended_vit": 0.9756}
    capped = {}
    for key, _l, _c in METHODS:
        rows = _parse_log(LOGS / f"{key}.log")
        if rows:
            capped[key] = max(r[4] for r in rows)
    full_cnn = _parse_log(LOGS / "cnn_full.log")
    if full_cnn:
        FULL["cnn"] = max(r[4] for r in full_cnn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    labels = [l for k, l, _ in METHODS]
    keys = [k for k, _l, _c in METHODS]
    x = np.arange(len(keys))
    cap_vals = [capped.get(k, np.nan) for k in keys]
    full_vals = [FULL.get(k, np.nan) for k in keys]
    ax1.bar(x - .2, cap_vals, .38, label="300 / class", color="#8AA9CE")
    ax1.bar(x + .2, full_vals, .38, label="Full corpus", color="#2b4c7e")
    for xi, v in zip(x - .2, cap_vals):
        if not np.isnan(v):
            ax1.text(xi, v + .015, f"{v:.4f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + .2, full_vals):
        if not np.isnan(v):
            ax1.text(xi, v + .015, f"{v:.4f}", ha="center", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace(" (ours)", "\n(ours)") for l in labels], fontsize=8.5)
    ax1.set_ylabel("Validation accuracy"), ax1.set_ylim(0, 1.15)
    ax1.set_title("(a) Accuracy by method and data budget", fontsize=11, loc="left")
    ax1.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.02))

    for key, label, color in METHODS:
        if key in capped and key in FULL:
            ax2.plot([0, 1], [capped[key], FULL[key]], "-o", color=color, ms=6, lw=2, label=label)
            delta = (FULL[key] - capped[key]) * 100
            ax2.annotate(f"{delta:+.2f} pts", (1, FULL[key]), textcoords="offset points",
                         xytext=(9, -3), fontsize=8.5, color=color)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["300 / class\n(70,484 img)", "Full corpus\n(681,309 img)"], fontsize=9)
    ax2.set_xlim(-.15, 1.45)
    ax2.set_ylabel("Validation accuracy")
    ax2.set_title("(b) Data efficiency: gain from 9.7× more data", fontsize=11, loc="left")
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right")
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=.25, lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig7_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig8_attention():
    """CLS-token attention over the 7x7 token grid, overlaid on the glyph.

    Runs the encoder manually rather than through nn.TransformerEncoder so the per-layer
    attention weights are reachable; asserts the manual pass reproduces the model's own
    logits, which doubles as a check on the equations written up in Section 3.4.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import sys
    sys.path.insert(0, str(OUT.parent))
    from train_merged import build_transforms, stratified_split
    from train_arch import build_for_eval

    ckpt_path = OUT.parent / "checkpoints" / "extended_vit_merged.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_for_eval(ckpt).eval()
    model.load_state_dict(ckpt["state_dict"])
    names = ckpt["class_names"]

    samples, _ = _manifest()
    _, _, test = stratified_split(samples)
    picks = [test[i] for i in range(0, len(test), max(1, len(test) // 6))][:6]

    _, eval_tf = build_transforms()

    def forward_with_attention(x):
        feat = model.forward_backbone(x)
        z = feat.flatten(2).transpose(1, 2)
        z = torch.cat([model.cls_token.expand(x.shape[0], -1, -1), z], 1) + model.pos_embed
        attns = []
        for layer in model.transformer.layers:          # post-norm, matching PyTorch's default
            a, w = layer.self_attn(z, z, z, need_weights=True, average_attn_weights=True)
            z = layer.norm1(z + layer.dropout1(a))
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(z))))
            z = layer.norm2(z + layer.dropout2(ff))
            attns.append(w)
        return model.head(model.norm(z[:, 0])), attns

    fig, axes = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.6))
    with torch.no_grad():
        for i, (path, label) in enumerate(picks):
            img = Image.open(path)
            x = eval_tf(img).unsqueeze(0)
            logits, attns = forward_with_attention(x)
            assert torch.allclose(logits, model(x), atol=1e-4), "manual forward diverged"

            pred = int(logits.argmax())
            # CLS row of the last layer, dropping the CLS->CLS entry, back onto the 7x7 grid
            cls_attn = attns[-1][0, 0, 1:].reshape(1, 1, 7, 7)
            heat = F.interpolate(cls_attn, size=(224, 224), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
            disp = (x[0] * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1).numpy()

            axes[0, i].imshow(disp)
            axes[0, i].set_title(f"{names[label]}", fontsize=15, pad=5,
                                 fontfamily=["Kohinoor Bangla", "Bangla MN", "sans-serif"])
            axes[1, i].imshow(disp)
            axes[1, i].imshow(heat, cmap="inferno", alpha=.55)
            axes[1, i].set_xlabel("✓" if pred == label else f"✗ → {names[pred]}",
                                  fontsize=11, color="#55A868" if pred == label else "#C44E52")
            for r in (0, 1):
                axes[r, i].set_xticks([]), axes[r, i].set_yticks([])
    axes[0, 0].set_ylabel("Input", fontsize=9.5)
    axes[1, 0].set_ylabel("CLS attention", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_attention.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig9_efficiency():
    """Accuracy against single-image CPU latency; bubble area is parameter count.

    Latency depends only on architecture, not on trained weights, so this can be measured
    for ViT-B/16 before its run finishes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time
    import torch
    import sys
    sys.path.insert(0, str(OUT.parent))
    from train_arch import EVAL_BUILDERS

    acc = {}
    for key, _l, _c in METHODS:
        rows = _parse_log(LOGS / f"{key}.log")
        if rows:
            acc[key] = max(r[4] for r in rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    torch.set_num_threads(1)  # one thread: comparable, and typical of a constrained deployment
    # label offsets tuned per method so the annotations don't collide with the bubbles
    OFFSETS = {"cnn": (0, -30), "resnet18": (-4, -46), "extended_vit": (2, 26),
               "vit": (0, -52)}
    for key, label, color in METHODS:
        if key not in acc:
            continue
        model = EVAL_BUILDERS[key](256).eval()
        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            for _ in range(3):
                model(x)
            t0 = time.perf_counter()
            for _ in range(15):
                model(x)
        ms = (time.perf_counter() - t0) / 15 * 1000

        # sqrt scaling: a linear area map spans 1,400x across these models and is unreadable
        ax.scatter(ms, acc[key], s=(params ** 0.5) / 25, color=color, alpha=.5,
                   edgecolor=color, lw=1.5, zorder=3)
        ax.annotate(f"{label}\n{params/1e6:.1f}M · {ms:.0f} ms",
                    (ms, acc[key]), textcoords="offset points", xytext=OFFSETS.get(key, (0, -34)),
                    ha="center", fontsize=8.5, color=color, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(2.6, 130)
    ax.set_ylim(0.08, 1.14)
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Single-image CPU latency, 1 thread (ms, log scale)")
    ax.set_ylabel("Validation accuracy (300 / class)")
    ax.set_title("Accuracy against inference cost — bubble area ∝ √parameters",
                 fontsize=11, loc="left", pad=14)
    ax.grid(alpha=.25, lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig1_workflow, fig2_dataset, fig3_architecture, fig4_samples,
               fig5_preprocessing, fig6_training_curves, fig7_results, fig8_attention,
               fig9_efficiency):
        try:
            fn()
            print(f"{fn.__name__}: ok")
        except Exception as e:  # one broken figure shouldn't block the others
            print(f"{fn.__name__}: FAILED — {type(e).__name__}: {e}")
