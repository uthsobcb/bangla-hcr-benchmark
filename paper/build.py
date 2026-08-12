"""Render draft.md to .docx (pandoc) and .pdf (LibreOffice), with figures embedded.

    ./venv/bin/python paper/build.py            # working draft, keeps drafting notes
    ./venv/bin/python paper/build.py --clean    # submission-shaped: notes and TODO sections stripped
    ./venv/bin/python paper/build.py --no-pdf

Figures are inserted after the paragraph that first mentions them, so the markdown itself stays
free of image tags.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).parent
DRAFT = OUT / "draft.md"

FIGURES = {
    1: ("fig1_workflow.png", "Overall workflow: three corpora, grapheme-level label reconciliation, "
                             "one shared split, four architectures, one test set."),
    2: ("fig2_dataset.png", "Dataset composition. (a) Per-class frequency, log scale. "
                            "(b) Images contributed by each corpus."),
    3: ("fig3_architecture.png", "ExtendedViT: a ResNet18 tokenizer feeding a four-layer Transformer "
                                 "encoder, with tensor shapes at each stage."),
    4: ("fig4_samples.png", "The same three characters as collected by each corpus. Note the "
                            "resolution difference and MatriVasha's inverted ink polarity."),
    5: ("fig5_preprocessing.png", "One glyph per corpus through polarity normalisation and three "
                                  "augmentation draws."),
    6: ("fig6_training_curves.png", "Validation accuracy and loss per epoch at the equal-budget "
                                    "setting; inset resolves the three pretrained models."),
    7: ("fig7_results.png", "Accuracy by method: 300 images per class (left bar) and the full "
                            "corpus (right). The warm-started run is hatched and excluded."),
    8: ("fig8_attention.png", "Classification-token attention over the 7x7 token grid, overlaid on "
                              "the input glyph. Attention avoids the shared matra head-line."),
    9: ("fig9_efficiency.png", "Accuracy against single-image CPU latency; bubble area is "
                               "proportional to the square root of parameter count."),
    10: ("fig10_data_efficiency_sweep.png", "Ekush-only budget sweep, ImageNet init throughout. "
                                            "(a) Accuracy differences fall within seed noise. "
                                            "(b) Calibration error before temperature scaling."),
    11: ("fig11_cross_source.png", "Cross-corpus generalisation collapses to chance on the classes "
                                   "each pair shares."),
}

# Sections that exist to coordinate the work, not to be read by a reviewer.
DROP_SECTIONS = ["## Writing still required", "## Outstanding experiments before submission",
                 "### Figures already built"]


def strip_drafting_apparatus(md):
    md = re.sub(r"\*\*Status of this draft\.\*\*.*?(?=\n---)", "", md, flags=re.S)
    md = re.sub(r"\*\*Note on the revision\.\*\*.*?(?=\n---)", "", md, flags=re.S)
    md = re.sub(r"^> .*$\n?", "", md, flags=re.M)          # blockquoted drafting notes
    md = re.sub(r"\n\*\(Elsevier and several.*?\)\*\n", "\n", md, flags=re.S)
    for heading in DROP_SECTIONS:
        # drop from the heading to the next heading of the same or higher level
        level = len(heading) - len(heading.lstrip("#"))
        pattern = re.escape(heading) + r".*?(?=\n#{1," + str(level) + r"} |\Z)"
        md = re.sub(pattern, "", md, flags=re.S)
    return re.sub(r"\n{4,}", "\n\n\n", md)


def insert_figures(md):
    """Place each figure after the paragraph that first mentions it."""
    paragraphs = md.split("\n\n")
    placed = set()
    out = []
    for para in paragraphs:
        out.append(para)
        if para.lstrip().startswith(("|", "```")):        # don't split a table or code block
            continue
        for n in sorted(set(int(m) for m in re.findall(r"Figure (\d+)", para))):
            if n in placed or n not in FIGURES:
                continue
            fname, caption = FIGURES[n]
            if (OUT / fname).exists():
                out.append(f"![Figure {n}. {caption}]({fname})")
                placed.add(n)
    missing = [n for n in FIGURES if n not in placed]
    if missing:
        print(f"  note: figures never referenced in text, appended at end: {missing}")
        out.append("## Figures")
        for n in missing:
            fname, caption = FIGURES[n]
            out.append(f"![Figure {n}. {caption}]({fname})")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true", help="strip drafting notes and coordination sections")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    if not shutil.which("pandoc"):
        sys.exit("pandoc not found — brew install pandoc")

    md = DRAFT.read_text()
    if args.clean:
        md = strip_drafting_apparatus(md)
    md = insert_figures(md)

    stem = "draft_clean" if args.clean else "draft"
    tmp = OUT / f"_{stem}_build.md"
    tmp.write_text(md)
    docx = OUT / f"{stem}.docx"

    # resource-path so the fig*.png references resolve; Bengali needs a font Word/LO can find
    cmd = ["pandoc", str(tmp), "-o", str(docx), "--resource-path", str(OUT),
           "--toc", "--toc-depth=2", "-V", "mainfont=Kohinoor Bangla",
           "--metadata", "title=" + re.search(r"^# (.+)$", md, re.M).group(1)]
    subprocess.run(cmd, check=True)
    print(f"wrote {docx.name}  ({docx.stat().st_size/1e6:.1f} MB)")

    title = re.search(r"^# (.+)$", md, re.M).group(1)
    if not args.no_pdf:
        build_pdf(tmp, stem, title)
    tmp.unlink()


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def build_pdf(md_path, stem, title):
    """HTML via pandoc, then headless Chrome to print it.

    LibreOffice would be the obvious route but is not installed here (the `soffice` on PATH is a
    dangling wrapper). Chrome renders Bengali correctly, which matters — the class labels and
    confusion tables are full of conjuncts.
    """
    html = OUT / f"_{stem}.html"
    css = OUT / "_paper.css"
    css.write_text("""
body { font-family: "Helvetica Neue", Arial, "Kohinoor Bangla", sans-serif;
       max-width: 46em; margin: 2em auto; line-height: 1.5; font-size: 11pt; color: #111; }
h1 { font-size: 20pt; line-height: 1.25; } h2 { font-size: 15pt; margin-top: 1.6em; }
h3 { font-size: 12.5pt; } img { max-width: 100%; display: block; margin: 1.2em auto; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 1em 0; }
th, td { border: 1px solid #bbb; padding: 4px 7px; text-align: left; }
th { background: #f0f0f0; } code { font-size: 9.5pt; background: #f6f6f6; padding: 1px 3px; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 1em; color: #555; }
@media print { h2 { page-break-after: avoid; } table, img { page-break-inside: avoid; } }
""")
    subprocess.run(["pandoc", str(md_path), "-o", str(html), "--standalone",
                    "--embed-resources", "--resource-path", str(OUT),
                    "--toc", "--toc-depth=2", "--css", str(css),
                    "--metadata", "title=" + title], check=True)

    if not Path(CHROME).exists():
        print(f"skip pdf: no PDF engine found (wrote {html.name}; open and print to PDF)")
        return
    pdf = OUT / f"{stem}.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()],
                   check=True, capture_output=True, timeout=300)
    print(f"wrote {pdf.name}  ({pdf.stat().st_size/1e6:.1f} MB)")
    html.unlink(), css.unlink()


if __name__ == "__main__":
    main()
