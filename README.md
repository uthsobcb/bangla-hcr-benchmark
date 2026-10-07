# Bengali character recognition — 4-method comparison

Four architectures trained on a merged 256-class set (RAS-Compound + Ekush + MatriVasha,
681,309 images), evaluated on one shared held-out split.

| method | what it is |
|---|---|
| `cnn` | from-scratch separable-conv net (62K params) — the no-pretraining baseline |
| `resnet18` | ImageNet-pretrained ResNet18, fine-tuned (11.3M) |
| `vit` | ImageNet-pretrained ViT-Base/16 via timm, fine-tuned (86.3M) |
| `extended_vit` | hybrid: ResNet18 backbone tokenizes into a Transformer encoder (23.9M) |

## Setup

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

Datasets are not in git. Expected layout:

```
data/
  RAS-Compound-character-dataset/{train,test}/<class_id>/*.png
  ekush-dataset/{<folder>/*.jpg, metaData_img.csv}
  MatriVasha_Dataset/{male,feamale}/<folder>/*.jpg
```

Folder→character maps: `ras_class_mapping.csv`, `matrivasha_mapping.csv`, and Ekush's own
`metaData_img.csv`. Characters are NFC-normalized and unioned into 256 classes, so the same
character from different datasets collapses to one label.

## Train

```bash
./venv/bin/python train_arch.py --arch cnn           # full data, 8 epochs
./venv/bin/python train_arch.py --arch vit --limit-per-class 300
./venv/bin/python train_arch.py --arch extended_vit --resume
```

Every arch shares `train_merged.py`'s manifest, transforms, and seed-42 stratified split, so
the train/val/test sets are byte-identical across methods and the numbers are comparable.

- `--limit-per-class N` caps samples per class **before** splitting and writes to
  `checkpoints/{arch}_merged_lpcN.pth`, so a capped comparison never overwrites full-data runs.
- `--resume` continues from `{arch}_merged*_last.pth`. A fresh run refuses to overwrite an
  existing best checkpoint — pass `--force` if you really mean to retrain from scratch.
- `--lr` overrides the per-arch default (ViT-Base needs 1e-4; 3e-4 diverges).

Rough cost on an M-series Mac (MPS), 8 epochs:

| | full 681K | capped 300/class |
|---|---|---|
| cnn | ~3 h | ~15 min |
| resnet18 | ~6 h | ~1 h |
| extended_vit | ~10–13 h | ~1.5 h |
| vit | ~70 h | ~7.5 h |

## Evaluate

```bash
./venv/bin/python evaluate.py                                    # every full-data checkpoint
./venv/bin/python evaluate.py --limit-per-class 300 --confusion  # the capped 4-way table
./venv/bin/python evaluate.py --arch cnn vit --per-class
```

Scores every checkpoint on the same test split and writes `results.csv`: accuracy, CER,
macro-F1, params, checkpoint size, single-image CPU latency, and **per-source accuracy**
(RAS / Ekush / MatriVasha) — the column that shows whether merging helped or just averaged.

CER equals top-1 error here: one label per image, single characters.

`--confusion` writes `confusion_{arch}.png`; `--per-class` prints sklearn's per-class report.

## Predict a word

```bash
./venv/bin/python predict_word.py word.png --model checkpoints/extended_vit_merged.pth
./venv/bin/python predict_word.py --selftest
```

Segments a word by vertical ink projection (splitting on the শিরোরেখা when the profile shows
one blob), classifies each crop, and gates sub-threshold predictions to `?` via `--min-conf`.
Loads any of the four architectures from the checkpoint's `arch` field. Always CPU.

## Paper

Two manuscript variants share `paper/body.tex`, `paper/references.bib` and the figures in `paper/`: `paper/ivc/` (Elsevier `elsarticle`, Image and Vision Computing) and `paper/ijdar/` (Springer Nature `sn-jnl`, IJDAR). Build either with `lualatex` + `bibtex` from its folder. Figures and notebooks regenerate from the saved
checkpoints and logs:

```bash
./venv/bin/python paper/figures.py           # all 11 figures -> paper/*.png
./venv/bin/python paper/make_notebooks.py    # build + execute notebooks 01-03
./venv/bin/python paper/make_notebooks.py 01 --no-exec   # build one, don't run it
```

The notebooks are the archival record — they re-run the analysis against saved checkpoints and
logs and keep every output inline, which the plain scripts do not. Training itself stays in
`train_arch.py`; the notebooks never retrain.

| notebook | contents |
|---|---|
| `01-dataset-analysis.ipynb` | label reconciliation, per-source composition, imbalance, fig2/fig4/fig5 files |
| `02-training-and-results.ipynb` | full training logs, results tables, fig6/fig7/fig9 files, confusion matrices |
| `03-attention-and-inference.ipynb` | architecture trace, CLS attention (fig8 file), CPU latency |

## Layout

```
train_merged.py        shared data pipeline (library): manifest, seed-42 split, transforms,
                       ExtendedViT, epoch loop. Every entry point imports it, which is what
                       keeps the architectures comparable.
train_arch.py          training CLI for all four architectures
evaluate.py            scoring on the shared split -> results/
cross_source.py        train on one corpus, test on another
temperature_scaling.py post-hoc calibration control
predict_word.py        word segmentation + inference
dashboard.py           live training dashboard at localhost:8765
progress.sh            one-shot / watch training status

data/                  the three corpora (not in git)
checkpoints/           trained weights (not in git)
logs/                  training logs (not in git)
results/               every generated table and confusion matrix (in git)
paper/                 draft, figures, build scripts
notebooks/             01-03 analysis notebooks, plus archived earlier experiments
```

`*_merged.pth` is the best checkpoint for a run; `*_last.pth` additionally stores optimiser
state and exists only so `--resume` works. Once a run is finished its `_last` file can be
deleted — together they account for most of `checkpoints/`.

## Cross-corpus and calibration experiments

```bash
./venv/bin/python cross_source.py --train ekush --test ras --arch resnet18
./venv/bin/python temperature_scaling.py --source ekush --limit-per-class 50
```

The first trains on one corpus and evaluates on another over the classes they share — the
experiment behind the paper's central finding. The second fits a single temperature on
validation logits and reports ECE before and after, which is the control that determines
whether a calibration difference is architectural.

## Manifest and splits

`manifest.csv.gz` lists all 681,309 images with `path` (relative to the data root), `source`, `label` (NFC grapheme), `class_idx`, and one column per experiment split: `split_full`, `split_lpc300`, `split_ekush_lpc{50,100,300,1000}` and `split_cross_{ras_to_ekush,ras_to_matrivasha,ekush_to_ras,matrivasha_to_ras}`. Values are `train`, `val`, `test` (`xtest` for cross-corpus targets) or empty. Regenerate with `python export_manifest.py` (SHA-256 of the output is printed and is reproducible).

## Licence

The code, label-reconciliation mapping files, and manifest in this repository are released under the [MIT Licence](LICENSE). The three source corpora (RAS-Compound, Ekush, MatriVasha) are not redistributed here and keep their own licences; check each source's terms before use.
