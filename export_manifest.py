"""Export the unified manifest and every train/val/test split used in the paper to one file.

Run on the machine that holds data/ (and trained the models), from the repo root:
    python export_manifest.py            # writes manifest.csv.gz and prints its SHA-256

One row per image. `split_*` columns hold train / val / test, or are empty when the image was
not used in that experiment (capped out by --limit-per-class, or outside the corpus). Cross-corpus
columns use xtest for the target corpus's evaluation images. Each split is produced by the same
stratified_split(seed=42) calls the training scripts make, so the file is the authoritative record
of the splits behind the reported numbers: anyone can use it instead of re-deriving them, since
the scan order of build_manifest() is filesystem-dependent.
"""
import csv
import gzip
import io
import hashlib
import sys

from train_merged import DATA, build_manifest, stratified_split
from train_arch import SOURCE_DIRS, restrict_to_source
from cross_source import split_by_source

OUT = "manifest.csv.gz"
SWEEP_BUDGETS = (50, 100, 300, 1000)              # Ekush-only sweep
CROSS_PAIRS = (("ras", "ekush"), ("ras", "matrivasha"), ("ekush", "ras"), ("matrivasha", "ras"))
CROSS_LIMIT = 300                                  # cross_source.py default


def labelled(*parts):
    """{path: split name} for (train, val, test) lists of (path, label)."""
    return {p: name for name, part in zip(("train", "val", "test"), parts) for p, _ in part}


def source_of(path):
    return next(k for k, d in SOURCE_DIRS.items() if d in path)


def main():
    samples, classes = build_manifest()
    columns = {}

    columns["split_full"] = labelled(*stratified_split(samples))
    columns["split_lpc300"] = labelled(*stratified_split(samples, limit_per_class=300))

    ekush, _ = restrict_to_source(samples, classes, "ekush")
    for n in SWEEP_BUDGETS:
        columns[f"split_ekush_lpc{n}"] = labelled(*stratified_split(ekush, limit_per_class=n))

    for train_src, test_src in CROSS_PAIRS:
        tr_all, te_all, _ = split_by_source(samples, classes, train_src, test_src)
        tr, va, _ = stratified_split(tr_all, limit_per_class=CROSS_LIMIT)
        col = labelled(tr, va, [])
        col.update({p: "xtest" for p, _ in te_all})
        columns[f"split_cross_{train_src}_to_{test_src}"] = col

    # sanity check against the sizes reported in the paper (Section 3.3 and Table 6)
    expected = {"split_full": (545289, 68010, 68010), "split_lpc300": (56414, 7035, 7035)}
    for name, want in expected.items():
        got = tuple(sum(v == s for v in columns[name].values()) for s in ("train", "val", "test"))
        print(f"{name}: train/val/test = {got}")
        if got != want:
            sys.exit(f"{name} does not match the paper's {want}: this scan order or data differs "
                     "from the one used for the reported results. Not writing a manifest.")

    header = ["path", "source", "label", "class_idx", *columns]
    with gzip.GzipFile(OUT, "wb", mtime=0) as gz, io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:  # mtime=0: reproducible hash
        w = csv.writer(f)
        w.writerow(header)
        for path, idx in sorted(samples):
            rel = path.split(str(DATA) + "/", 1)[-1]
            w.writerow([rel, source_of(path), classes[idx], idx, *(c.get(path, "") for c in columns.values())])

    print(f"wrote {OUT}: {len(samples)} images, {len(columns)} split columns")
    print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
