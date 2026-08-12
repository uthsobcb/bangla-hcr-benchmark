"""Build and execute the paper's notebooks, so results and logs stay saved inside them.

    ./venv/bin/python paper/make_notebooks.py            # build + execute all
    ./venv/bin/python paper/make_notebooks.py --no-exec  # build only (fast)
    ./venv/bin/python paper/make_notebooks.py 01 03      # only these

The .py modules stay the source of truth for training; these notebooks are the record —
they re-run analysis against the saved checkpoints and logs and keep every output inline.
"""
import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# ─────────────────────────────────────────────────────────────── 01 dataset ──
NB01 = [
    md("""
# 01 · Unified Dataset Construction and Analysis

Builds the merged 256-class corpus from RAS-Compound, Ekush and MatriVasha, and reproduces
every dataset number quoted in the paper (Section 3.2) along with Figures 2, 4 and 5.
"""),
    code("""
import sys, os
sys.path.insert(0, os.path.abspath('..'))
%matplotlib inline
from collections import Counter, defaultdict
import statistics
from train_merged import build_manifest, stratified_split, load_mapping, ROOT, DATA
print('project root:', ROOT)
"""),
    md("""
## Label reconciliation

Each corpus indexes classes by a local folder number whose meaning is defined only by that
project's own mapping file. We resolve every folder to its Bengali grapheme, NFC-normalise,
and take the union — so the same character from different corpora collapses to one label.
"""),
    code("""
ras   = load_mapping(ROOT/'ras_class_mapping.csv', 'class_id', 'bengali_character')
ekush = load_mapping(DATA/'ekush-dataset'/'metaData_img.csv', 'Folder Name', 'Char Name')
matri = load_mapping(ROOT/'matrivasha_mapping.csv', 'folder', 'character')

R, E, M = set(ras.values()), set(ekush.values()), set(matri.values())
print(f'RAS-Compound : {len(R)} classes')
print(f'Ekush        : {len(E)} classes')
print(f'MatriVasha   : {len(M)} classes')
print(f'Union        : {len(R|E|M)} classes')
"""),
    code("""
print('Pairwise overlap')
print(f'  RAS  n Ekush : {len(R&E)}')
print(f'  RAS  n Matri : {len(R&M)}')
print(f'  Ekush n Matri: {len(E&M)}')
print(f'  all three    : {len(R&E&M)}')
print()
print('Contributed by exactly one source')
print(f'  RAS only     : {len(R-E-M)}')
print(f'  Ekush only   : {len(E-R-M)}')
print(f'  Matri only   : {len(M-R-E)}')
print(f'  total unique : {len(R-E-M) + len(E-R-M) + len(M-R-E)} of {len(R|E|M)}')
"""),
    md("## Manifest and per-source composition (Table 1)"),
    code("""
samples, classes = build_manifest()
src = Counter('RAS-Compound' if 'RAS-Comp' in p else
              'Ekush' if 'ekush' in p else 'MatriVasha' for p, _ in samples)
for name, n in src.most_common():
    print(f'{name:14s} {n:>7,} images')
print(f'{"TOTAL":14s} {len(samples):>7,} images across {len(classes)} classes')
"""),
    md("""
## Class imbalance

The distribution is **not** Zipfian but sharply bimodal: a broad flat body, then a cliff to a
27-class rare tail contributed entirely by RAS-Compound.
"""),
    code("""
counts = Counter(l for _, l in samples)
v = sorted(counts.values())
print(f'min {v[0]}  max {v[-1]}  median {statistics.median(v):.0f}  mean {statistics.mean(v):.0f}')
print(f'classes with <100 samples: {sum(1 for x in v if x < 100)}')
print(f'classes with <500 samples: {sum(1 for x in v if x < 500)}')

rare = [classes[c] for c, n in counts.items() if n < 100]
print(f'\\nrare classes ({len(rare)}): ' + ' '.join(rare))
"""),
    md("## Splits — identical for every architecture (seed 42)"),
    code("""
for name, lim in [('Full corpus', None), ('Equal budget (300/class)', 300)]:
    tr, va, te = stratified_split(samples, limit_per_class=lim)
    print(f'{name:26s} train {len(tr):>7,} | val {len(va):>6,} | test {len(te):>6,}')
"""),
    md("## Figures 2, 4 and 5"),
    code("""
sys.path.insert(0, os.path.abspath('../paper'))
from figures import fig2_dataset, fig4_samples, fig5_preprocessing
for fn in (fig2_dataset, fig4_samples, fig5_preprocessing):
    fn(); print(fn.__name__, 'written')
"""),
    code("""
from IPython.display import Image, display
for f in ('fig2_dataset', 'fig4_samples', 'fig5_preprocessing'):
    display(Image(f'../paper/{f}.png'))
"""),
]

# ────────────────────────────────────────────────────────── 02 training/results ──
NB02 = [
    md("""
# 02 · Training Runs and Results

The training itself runs via `train_arch.py` (hours per model); this notebook is the record.
It reproduces the full logs, the results tables of Section 5, and Figures 6, 7 and 9.
"""),
    code("""
import sys, os, re
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../paper'))
%matplotlib inline
from pathlib import Path
import pandas as pd
LOGS = Path('../logs')
print('logs found:', sorted(p.name for p in LOGS.glob('*.log')))
"""),
    md("""
## Training configuration

Every architecture shares `train_merged.py`'s manifest, transforms and seed-42 split, so all
runs see byte-identical data. Only the model varies.
"""),
    code("""
from train_arch import ARCHS
for name, spec in ARCHS.items():
    print(f'{name:14s} default lr {spec["lr"]}')
print('''
Shared: AdamW (wd 1e-4), cross-entropy w/ label smoothing 0.1,
cosine anneal to 1e-6 over 8 epochs, batch 32, grad-clip 1.0,
backbone lr = 0.1 x base for pretrained models.''')
"""),
    md("## Complete training logs"),
    code("""
RUNS = [('cnn', '300/class'), ('resnet18', '300/class'), ('extended_vit', '300/class'),
        ('vit', '300/class'), ('extended_vit_imagenet', '300/class, ImageNet init'),
        ('cnn_full', 'full corpus')]
for stem, cond in RUNS:
    p = LOGS / f'{stem}.log'
    print('=' * 78)
    print(f'{stem}  ({cond})')
    print('=' * 78)
    if not p.exists():
        print('(not run)\\n'); continue
    for line in p.read_text(errors='replace').splitlines():
        if any(k in line for k in ('Epoch [', 'Best Val', 'Test Acc', 'Total params',
                                   'Warm-started', 'Device', 'Train ', 'images,')):
            print(line)
    print()
"""),
    md("## Results tables"),
    code("""
for csv, title in [('../results_lpc300.csv', 'Equal budget (300 per class)'),
                   ('../results_lpc300_imagenet.csv', 'Equal budget, ImageNet-init ablation'),
                   ('../results_full.csv', 'Full corpus')]:
    print(f'\\n### {title}')
    try:
        print(pd.read_csv(csv).to_string(index=False))
    except FileNotFoundError:
        print('(evaluate.py has not produced this yet)')
"""),
    md("## Figures 6, 7 and 9"),
    code("""
from figures import fig6_training_curves, fig7_results, fig9_efficiency
for fn in (fig6_training_curves, fig7_results, fig9_efficiency):
    fn(); print(fn.__name__, 'written')
"""),
    code("""
from IPython.display import Image, display
for f in ('fig6_training_curves', 'fig7_results', 'fig9_efficiency'):
    display(Image(f'../paper/{f}.png'))
"""),
    md("## Confusion matrices"),
    code("""
from IPython.display import Image, display
import glob
found = sorted(glob.glob('../paper/confusion_*.png'))
print(found if found else '(produced by evaluate.py --confusion)')
for f in found:
    display(Image(f))
"""),
]

# ──────────────────────────────────────────────────────────── 03 attention ──
NB03 = [
    md("""
# 03 · Attention Analysis and CPU Inference

Where ExtendedViT looks, and what it costs to run. Reproduces Figure 8 and the latency
numbers of Section 5.4.
"""),
    code("""
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../paper'))
%matplotlib inline
import torch
from train_arch import build_for_eval
ckpt = torch.load('../checkpoints/extended_vit_merged.pth', map_location='cpu', weights_only=False)
model = build_for_eval(ckpt).eval()
model.load_state_dict(ckpt['state_dict'])
print('arch:', ckpt.get('arch'), '| classes:', len(ckpt['class_names']))
print('params:', f"{sum(p.numel() for p in model.parameters()):,}")
"""),
    md("""
## Architecture

A 224x224 input becomes a 512x7x7 map, flattened to 49 tokens of dimension 512; a CLS token
and learned positional embeddings are added, and 4 post-norm encoder layers attend over the
resulting 50 tokens. Classification reads the CLS token.
"""),
    code("""
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    feat = model.forward_backbone(x)
    tokens = feat.flatten(2).transpose(1, 2)
    seq = torch.cat([model.cls_token, tokens], 1) + model.pos_embed
print('input          ', tuple(x.shape))
print('backbone map   ', tuple(feat.shape))
print('flattened      ', tuple(tokens.shape))
print('+CLS +pos      ', tuple(seq.shape))
print('logits         ', tuple(model(x).shape))
print()
for n, m in [('backbone', [model.conv1, model.bn1, model.layer1, model.layer2,
                           model.layer3, model.layer4]),
             ('transformer', [model.transformer]), ('head', [model.norm, model.head])]:
    print(f'{n:12s} {sum(p.numel() for mm in m for p in mm.parameters()):>12,} params')
"""),
    md("""
## Figure 8 — CLS attention

Attention concentrates on the conjunct body and avoids the মাত্রা head-line, which nearly every
Bengali character shares and which therefore carries little class-discriminative information.
The manual encoder pass is asserted to reproduce the model's own logits.
"""),
    code("""
from figures import fig8_attention
fig8_attention(); print('written')
"""),
    code("""
from IPython.display import Image, display
display(Image('../paper/fig8_attention.png'))
"""),
    md("## CPU inference cost"),
    code("""
import time, warnings
warnings.filterwarnings('ignore')  # timm imports tqdm.auto, which warns about ipywidgets
from train_arch import EVAL_BUILDERS
torch.set_num_threads(1)
x = torch.randn(1, 3, 224, 224)
print(f"{'model':16s} {'params':>12s} {'ms/image':>10s}")
for name, build in EVAL_BUILDERS.items():
    m = build(256).eval()
    with torch.no_grad():
        for _ in range(3): m(x)
        t0 = time.perf_counter()
        for _ in range(15): m(x)
    ms = (time.perf_counter() - t0) / 15 * 1000
    print(f'{name:16s} {sum(p.numel() for p in m.parameters()):>12,} {ms:>10.1f}')
"""),
]

NOTEBOOKS = {"01": ("01-dataset-analysis.ipynb", NB01),
             "02": ("02-training-and-results.ipynb", NB02),
             "03": ("03-attention-and-inference.ipynb", NB03)}


def build(key, execute):
    filename, cells = NOTEBOOKS[key]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    path = NB_DIR / filename

    if execute:
        from nbclient import NotebookClient
        # cwd = notebooks/, matching the relative paths the cells use
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()

    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("which", nargs="*", default=list(NOTEBOOKS), choices=list(NOTEBOOKS) + [])
    p.add_argument("--no-exec", action="store_true", help="build without running the cells")
    args = p.parse_args()

    NB_DIR.mkdir(exist_ok=True)
    for key in args.which:
        try:
            path = build(key, execute=not args.no_exec)
            print(f"{key}: {path.name} {'built' if args.no_exec else 'executed'}")
        except Exception as e:  # one failing notebook shouldn't block the rest
            print(f"{key}: FAILED — {type(e).__name__}: {e}")
