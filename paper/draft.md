# Single-Corpus Accuracy Overstates Bengali Compound-Character Recognition: A Unified 256-Class Benchmark and Cross-Corpus Evaluation

**Status of this draft.** All experiments are complete and every number below is measured. Fields
marked `[FILL]` need information only the authors have (affiliations, funding, licences, venue).

**Note on the revision.** Restructured to lead with the benchmark and the cross-corpus finding,
carrying the architecture result inside them. An earlier version claimed a data-efficiency advantage
in accuracy; that claim came from runs whose backbones had inherited previously trained Bengali
weights, and survived neither the corrected initialisation nor seed repetition. The supported
architecture claim is about calibration, reported alongside the negative accuracy result.

---

## Title Page

**Title.** Single-Corpus Accuracy Overstates Bengali Compound-Character Recognition: A Unified
256-Class Benchmark and Cross-Corpus Evaluation

*Alternatives:* (a) Cross-Corpus Generalisation Fails at Chance on Bengali Compound Characters —
blunter, and the finding carries it; (b) A Unified 256-Class Bengali Compound-Character Benchmark
with Cross-Corpus Baselines — leads with the resource if the venue is dataset-oriented.

*Rejected:* any title foregrounding the four-architecture comparison. The architectures differ by
at most one accuracy point, and naming them promises a contribution the results do not deliver.

**Authors.** Uthsob Chakraborty`[FILL: full author list and order]`

**Affiliations.** `[FILL: department, institution, city, country for each author]`

**Corresponding author.** Uthsob Chakraborty, uthsob9@gmail.com `[FILL: confirm, add postal address]`

**ORCID.** `[FILL]`

**Keywords.** Bengali handwritten character recognition; compound characters (যুক্তাক্ষর); cross-corpus
generalisation; domain shift; benchmark; Vision Transformer; hybrid CNN–Transformer; model calibration

---

## Highlights

*(Elsevier and several other venues require 3–5 bullets, ≤85 characters each. Trim to the
venue's limit.)*

- A 256-class, 681,309-image Bengali benchmark unifying three corpora by Unicode grapheme
- Models transfer across corpora at chance: 0.97 same-corpus, 0.03 cross-corpus
- Single-corpus accuracy measures corpus familiarity, not character recognition
- Four architectures compared; none separates on accuracy once initialisation is controlled
- Apparent calibration gaps between architectures vanish under temperature scaling

---

## Abstract

*(~300 words, per outline)*

Bengali is written in an alpha-syllabary whose compound characters (যুক্তাক্ষর) are formed by
conjoining two or more basic consonants, producing an open class of more than 170 structurally
intricate, mutually confusable glyphs. Progress on recognising them is limited less by architecture
than by data: the available corpora are fragmented, collected under incompatible conventions, and
individually too small to support a large-scale study.

We therefore unify three publicly available Bengali handwriting corpora — RAS-Compound, Ekush, and
MatriVasha — into a single 256-class benchmark of 681,309 images. Labels are reconciled by Unicode
NFC normalisation of the underlying graphemes rather than by dataset-local folder indices, so the
same character collected by different projects collapses to one label while 159 classes contributed
by a single source are preserved. We characterise the result rather than merely assembling it: its
imbalance proves to be an artifact of source size rather than character frequency, and each
constituent corpus is internally near-balanced.

We benchmark four architectures on it under one protocol: a from-scratch CNN, a fine-tuned ImageNet
ResNet18, a fine-tuned ViT-B/16, and ExtendedViT — a hybrid in which an ImageNet-pretrained ResNet18
tokenizes the image into 49 tokens for a four-layer Transformer encoder. ViT-B/16 leads at an equal
data budget (0.9662); on the full corpus ResNet18 and ExtendedViT are indistinguishable (0.9764 vs
0.9760, 27 images in 68,010).

The benchmark's most consequential result concerns none of these architectures individually. Trained
on one constituent corpus and tested on another, restricted to the classes they share, every model
falls from 0.96–1.00 same-corpus accuracy to 0.02–0.05 cross-corpus — at or barely above chance, in
both directions, for both architectures. We verify this is domain shift rather than label
disagreement. Single-corpus accuracy therefore measures familiarity with a corpus's rendering
conventions rather than recognition of Bengali characters, which recasts merging as domain coverage
rather than dataset enlargement.

The architectures themselves separate remarkably little. A controlled single-corpus sweep over
per-class budgets from 50 to 1,000, repeated across three seeds, finds no significant accuracy
difference between a convolutional baseline and a CNN–Transformer hybrid at any budget. The hybrid
does show roughly half the baseline's expected calibration error — but temperature scaling, a single
fitted scalar, removes that difference entirely and leaves the scaled baseline better calibrated
than the unscaled hybrid, so it is not an architectural property. We report three further
methodological cautions arising from this benchmark: warm-starting from a previously trained
checkpoint, routine practice in this literature, inflated our own result by 1.43 accuracy points;
macro-F1 does not measure long-tail performance when class rarity is confounded with source; and
uncorrected calibration comparisons between architectures should not be trusted.

---

## 1. Introduction

Bengali is the seventh most spoken language in the world, with approximately 284 million speakers
[Ethnologue, 27th ed.]. Its script is an abugida of 11 vowels and 39 consonants, 10 numerals, and an
open set of compound characters formed by conjoining two or more consonants. Counting diacritics and
modifiers, roughly 13,000 grapheme variations are possible [30], [32], of which more than 170 are
conventionally treated as distinct conjunct classes [7].

Handwritten recognition of this script is hard for reasons that recur across the literature: the
cursive connecting head-line (মাত্রা) that binds characters into a visual unit; very high inter-class
similarity, where two distinct conjuncts may differ by a single stroke or dot; wide variation in
individual writing style; and a frequency distribution over characters that approximately follows
Zipf's law, leaving many compound classes represented by only a handful of samples [33].

Compound characters concentrate all of these difficulties. Classical feature-engineering approaches
reached only 79–87% on compound benchmarks [2]–[5], and although deep convolutional networks lifted
this to 90–99% depending on dataset balance [9], [16], [23], the reported ceiling is strongly
dependent on how balanced and how large the evaluation set is. The Vision Transformer [22] has since
been shown to outperform VGG-16 and ResNet-50 on Bengali *basic* characters — 98.26% against 94.54%
and 93.12% respectively [17] — but its application to compound characters remains sparse, and Parvez
et al. explicitly identify this as a critical research gap.

Two considerations complicate a naive transfer of ViT to this problem. First, ViT lacks the
translation equivariance and locality that convolutions encode as architectural priors, and
consequently underperforms comparably sized ResNets when training data is merely mid-sized rather
than web-scale [22]. Bengali compound-character data is emphatically not web-scale. Second, the
available corpora are fragmented: each is collected under different conventions, at different
resolutions, with different and only partially overlapping class inventories, so no single one of
them supports a large-scale study.

This paper makes four contributions, and they are connected: the benchmark is what made the
remaining three measurable.

**A unified benchmark.** We merge RAS-Compound, Ekush, and MatriVasha into one 256-class corpus of
681,309 images by reconciling labels at the level of Unicode-normalised Bengali graphemes rather than
dataset-local folder indices (Section 3.2). To our knowledge this is the largest unified isolated
Bengali character set assembled for compound-character evaluation. We characterise it rather than
merely assembling it: its imbalance is an artifact of source size rather than character frequency
(Section 3.2), and — our most consequential finding — models trained on any one constituent corpus
transfer to another **at chance** (Section 5.5). No single Bengali corpus supports a model that
generalises, which recasts merging as domain coverage rather than dataset enlargement and casts
doubt on single-corpus accuracies reported throughout the literature.

**Four architectures under one protocol.** A from-scratch CNN, a fine-tuned ImageNet ResNet18, a
fine-tuned ViT-B/16, and our hybrid, all trained and evaluated on a byte-identical split at two
data budgets (Section 4). ViT-B/16 leads at an equal budget, reproducing on compound characters the
ordering Parvez et al. [17] reported for basic characters. Without pretraining, a CNN reaches 0.013
accuracy on the minority corpus that supplies 27 otherwise-unavailable classes — on a merged corpus
of unequal sources, transfer learning is what makes the minority source learnable at all.

**A hybrid architecture, evaluated and not sustained.** We include ExtendedViT, which uses an
ImageNet-pretrained ResNet18 as a learned tokenizer feeding a Transformer encoder (Section 3.4). A
controlled single-corpus sweep over per-class budgets from 50 to 1,000, repeated across seeds, finds
**no significant accuracy advantage at any budget**; an apparent two-fold calibration advantage
proves fully correctable by temperature scaling and is therefore not architectural (Section 5.4). We
report it as a baseline rather than a contribution. Its one durable property is efficiency: 17 ms
single-image CPU inference against ViT-B/16's 56 ms, for roughly one accuracy point less.

**Three methodological cautions,** each found by attempting to falsify a claim of our own.
*Initialisation:* both pretrained baselines inherited previously trained Bengali weights, routine
practice that inflated our equal-budget result by 1.43 points and moved first-epoch validation
accuracy from 0.234 to 0.703 (Section 5.8). *Macro-F1:* on this corpus the rarest classes score the
**highest** F1, because rarity is confounded with source image quality, so macro-F1 does not measure
long-tail performance (Section 5.10). *Calibration:* architecture-level ECE differences are
correctable post hoc and should always be reported temperature-scaled (Section 5.4).

---

## 2. Literature Review

> Insert the existing standalone review here (Sections 2.1–2.10 of
> `Literature_Review_Bengali_ViT_Compound_Characters.docx`), **plus the new subsection 2.7 drafted
> below**, renumbering the existing 2.7–2.10 to 2.8–2.11. The existing review covers classical
> approaches, CNN methods, ViT and transformer approaches, compound-character work, ImageNet
> transfer learning, benchmark datasets, reported accuracies, and the research gaps. It contains no
> coverage of cross-dataset evaluation or domain generalisation, which is now this paper's central
> finding — hence the addition.

### 2.7 Cross-Dataset Evaluation and Domain Generalisation *(new subsection to insert)*

> **Drafting note.** Citations marked ⚑ are not in the existing numbered scheme and must be added to
> the reference list. Verify each against the source before submission; the claims attributed to them
> are standard in the domain-generalisation literature but the specific numbers should be checked.

A recurring feature of the Bangla HCR literature is that models are trained and evaluated **within**
a single corpus, with the train/test split drawn from the same collection. Reported accuracies
therefore describe within-corpus competence. Several studies do report results across multiple
benchmarks — BanglaNet [35] across CMATERdb, BanglaLekha-Isolated and Ekush; CompoundDenseNet [23]
across the same three, at 96.2–98.5%; Borno [15] trained on roughly one million images spanning
several datasets — but in each case the model is trained and tested on each corpus separately. This
is multi-benchmark *reporting*, not a test of transfer: it establishes that an architecture works on
several datasets when trained on each, not that a model trained on one generalises to another.

Genuine held-out-corpus evaluation is rare. The clearest instance in the reviewed literature is
Raquib et al. [25], who train on their own 78-class dataset and evaluate on the external CHBCR
benchmark, reporting 96.49% against 98.84% in-domain — a drop of roughly two points, which would
suggest that cross-dataset transfer is largely intact. We are not aware of a systematic
train-on-one, test-on-another study across the major Bangla isolated-character corpora, and Section
5.5 of this paper reports a substantially different picture when one is run.

The wider computer-vision literature has long held that dataset-specific bias inflates within-dataset
performance. Torralba and Efros ⚑ showed that a classifier can identify which of several standard
datasets an image came from with high accuracy — a direct demonstration that datasets carry
signatures unrelated to their nominal content — and that models trained on one dataset degrade
markedly when evaluated on another purporting to cover the same categories. Recht et al. ⚑ found
measurable accuracy drops for ImageNet classifiers on a newly collected test set constructed to
follow the original protocol, indicating that even careful replication of a collection procedure
shifts the distribution. Domain adaptation theory ⚑ formalises this: generalisation error on a
target domain is bounded by source error plus a divergence term between the domains, so arbitrarily
low source error does not constrain target error when the divergence is large.

For handwriting specifically the sources of divergence are concrete and well understood: capture
device and resolution, binarisation and ink polarity, writing instrument, paper, and — most
importantly — the writer population, since each corpus recruits a distinct set of contributors with
their own regional and generational script conventions. Section 3.2 documents exactly these
differences among the three corpora merged here: RAS-Compound stores high-resolution binarised
white-on-black glyphs, Ekush is natively 28×28, and MatriVasha is black-on-white.

The gap this identifies is therefore methodological rather than architectural. If within-corpus
accuracy is the field's only reported metric, and if that metric substantially overstates
performance on newly collected handwriting, then a decade of reported improvements may partly
reflect increasingly effective fitting of corpus-specific signatures. Establishing whether this is
so requires held-out-corpus evaluation as a standard protocol, which — to our knowledge — has not
been systematically applied to Bangla compound characters. That is the gap Section 5.5 addresses.

> **Add to Section 2.9 (Research Gaps), as a seventh gap:** *No systematic cross-corpus evaluation
> exists for Bangla handwritten characters.* Multi-benchmark reporting is common, but
> train-on-one/test-on-another transfer is essentially unmeasured, leaving open whether reported
> accuracies reflect character recognition or corpus familiarity.

Two points from that review bear directly on the design that follows and are worth restating here.
Dosovitskiy et al. [22] show that ViT trails a comparably sized ResNet when trained on ImageNet-1k
alone, becomes comparable with ImageNet-21k pretraining, and only surpasses it at JFT-300M scale —
"large-scale pretraining trumps inductive bias." For a low-resource script, the corollary is that a
pure ViT is the *least* favourable architecture unless a strong prior is supplied externally. And the
hybrid designs that do exist for Bengali — ResViT [19], the EfficientNetB3–ViT–Conformer of Raquib et
al. [25] — combine CNN and transformer branches *in parallel* and fuse their outputs. ExtendedViT
instead composes them *serially*, using convolution to produce the token sequence that attention then
operates over, which is the hybrid variant described in the original ViT paper's appendix but not
previously evaluated on this script.

---

## 3. Methodology

### 3.1 Overall Workflow

**Figure 1** (`fig1_workflow.png`) shows the end-to-end pipeline: three source corpora →
grapheme-level label reconciliation → unified 256-class manifest → stratified split → polarity
normalisation and augmentation → four architectures trained under an identical protocol →
evaluation on one shared test set.

The pipeline is deliberately organised so that every architecture consumes byte-identical data.
Dataset scanning, label reconciliation, splitting, and image transformation are implemented once and
shared; the architecture is the only variable that changes between runs. This matters because the
comparison in Section 6 turns on differences of one to two accuracy points, which are easily
manufactured by an inconsistent split.

### 3.2 Dataset

We combine three corpora. **RAS-Compound** is a small, purpose-built compound-character set.
**Ekush** [Rabby et al., 2018] is among the largest isolated Bengali character collections, gathered
from 3,086 writers. **MatriVasha** is a compound-character corpus collected with per-writer gender
metadata, which we do not use as a label but which contributes writer diversity.

**Table 1. Composition of the unified corpus.**

| Source | Images | Classes | Classes unique to this source | Mean images/class |
|---|---:|---:|---:|---:|
| RAS-Compound | 7,830 | 119 | 27 | 66 |
| Ekush | 367,018 | 122 | 80 | 3,008 |
| MatriVasha | 306,461 | 115 | 52 | 2,665 |
| **Unified** | **681,309** | **256** | — | **2,661** |

Class inventories overlap only partially: 37 classes are shared between RAS-Compound and Ekush, 58
between RAS-Compound and MatriVasha, 8 between Ekush and MatriVasha, and just 3 appear in all three.
159 of the 256 classes are contributed by exactly one source. The merge is therefore not a
redundant enlargement of an existing set — it materially extends class coverage, which is the
stated need in the compound-character literature [23].

**Label reconciliation.** Each source indexes its classes by a local folder number whose meaning is
defined only by that project's own mapping file. Merging on folder index would conflate unrelated
characters. We instead resolve every folder to its Bengali grapheme via the source's mapping
(`ras_class_mapping.csv`, Ekush's `metaData_img.csv`, `matrivasha_mapping.csv`), apply Unicode NFC
normalisation, and take the union of the resulting grapheme set as the label space. NFC matters
because the same conjunct may be encoded with different code-point sequences across projects; without
normalisation, visually and semantically identical characters would occupy separate classes.

**Figure 2** (`fig2_dataset.png`) shows (a) the sorted per-class frequency curve on a log axis and
(b) the relative contribution of each source. **Figure 4** (`fig4_samples.png`) shows the same three
characters as collected by each corpus, and makes the heterogeneity concrete: RAS-Compound stores
high-resolution white-on-black binarised glyphs, Ekush is natively 28×28 and visibly pixelated when
upsampled, and MatriVasha is **black-on-white** — the opposite ink polarity. Any merge of these three
must reconcile that variation before a single model can consume them, which is the subject of the
next section.

**Class imbalance.** Class frequencies range from 59 to 5,740 images with a median of 2,636. The
distribution is not Zipfian, as might be expected from natural Bengali text frequency [33], but
sharply *bimodal*: 229 classes form a broad, comparatively flat body between roughly 2,000 and 5,700
samples, and the remaining 27 fall off a cliff to fewer than 100 (Figure 2a). Those 27 are exactly
the classes contributed uniquely by RAS-Compound, whose 7,830 images spread across 119 classes
average only 66 per class.

**The imbalance is an artifact of source size, not of character frequency.** Each corpus is
internally near-balanced: within Ekush, per-class counts run from 514 to 4,067 with an interquartile
range of just 3,056–3,079; within MatriVasha, 2,548–2,563; within RAS-Compound, a uniform 34–70. The
bimodality in Figure 2(a) therefore arises entirely from RAS-Compound being a small collection
(≈67 images per class) whose 27 unique classes no other corpus supplies — not from those conjuncts
being intrinsically rarer in written Bengali.

This distinction matters for interpreting any per-class result. Because class rarity and source
identity are perfectly confounded in this corpus — every low-frequency class is a RAS-Compound class,
and RAS-Compound has a distinctive image convention (high-resolution, cleanly binarised) — a model
that performs better on rare classes cannot be shown to handle *rarity* better rather than simply
handling *that corpus's image style* better. We therefore do not claim rare-class advantage for any
architecture, and we report macro-averaged F1 alongside accuracy as a descriptive statistic rather
than as evidence about the long tail. Establishing genuine data-efficiency behaviour requires
controlled subsampling within a single source (Section 7).

### 3.3 Preprocessing and Data Splitting

**Polarity normalisation.** As Figure 4 shows, the three sources do not agree on ink polarity:
RAS-Compound and Ekush store light ink on a dark ground, MatriVasha the inverse. Left unaddressed,
the same character would present to the network as two different visual patterns depending on its
source, and the model would be forced to learn each convention separately. We normalise by comparing
the mean intensity of the image border against the global mean and inverting when the border is
brighter, which yields bright ink on a dark ground regardless of source convention. This is applied
before all other transformations. **Figure 5** (`fig5_preprocessing.png`) traces one glyph per source
through normalisation and three augmentation draws; the MatriVasha row shows the inversion taking
effect.

**Transformations.** Images are converted to three-channel grayscale (the pretrained backbones expect
three channels) and resized to 224×224. Training additionally applies random rotation of ±15°,
random affine translation of up to 10% with 5° shear, and colour jitter of 0.3 in brightness and
contrast. Horizontal flipping is deliberately excluded: Bengali characters are direction-sensitive
and a mirrored glyph is not a valid sample of its own class. All images are normalised to zero mean
and unit variance per channel using μ = σ = 0.5.

**Splitting.** We use a stratified 80/10/10 split computed per class with a fixed seed, so that rare
classes are represented in all three partitions rather than concentrated in whichever partition
chance assigns them. Classes with fewer than three samples contribute entirely to training. The full
corpus yields 545,289 training, 68,010 validation, and 68,010 test images. The split is computed
once from the manifest and reused by every architecture; the test partition is never used for model
selection, which is performed on validation accuracy alone.

### 3.4 Proposed Method: ExtendedViT

ExtendedViT composes a convolutional backbone and a Transformer encoder serially, using the former
as a learned tokenizer for the latter.

**Convolutional tokenizer.** An input image $x \in \mathbb{R}^{3 \times 224 \times 224}$ is passed
through the convolutional stages of an ImageNet-pretrained ResNet18, up to and including `layer4`
and excluding the global pooling and classification head:

$$F = f_{\text{ResNet18}}(x) \in \mathbb{R}^{512 \times 7 \times 7}$$

**Tokenisation.** The spatial map is flattened along its spatial axes into a sequence of $N = 49$
tokens of dimension $D = 512$, where each token summarises a 32×32-pixel receptive region of the
input:

$$F' = \text{flatten}(F) \in \mathbb{R}^{49 \times 512}$$

A learnable classification token $x_{\text{cls}} \in \mathbb{R}^{1 \times D}$ is prepended and a
learnable positional embedding $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ is added:

$$z_0 = [\,x_{\text{cls}}\,;\,F'_1\,;\,F'_2\,;\,\dots\,;\,F'_{49}\,] + E_{\text{pos}}$$

Both are initialised from a truncated normal distribution with standard deviation 0.02.

**Transformer encoder.** The 50-token sequence passes through $L = 4$ post-norm encoder layers, each
comprising multi-head self-attention with $h = 8$ heads and a position-wise feed-forward network of
inner dimension 2048, with residual connections and layer normalisation:

$$z'_\ell = \text{LN}\big(z_{\ell-1} + \text{MSA}(z_{\ell-1})\big)$$
$$z_\ell = \text{LN}\big(z'_\ell + \text{FFN}(z'_\ell)\big), \qquad \ell = 1 \dots L$$

where multi-head self-attention is defined as usual over $h$ heads of dimension $d_k = D/h = 64$:

$$\text{MSA}(z) = \big[\text{head}_1 ; \dots ; \text{head}_h\big] W^O, \qquad
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}}\right) V_i$$

with $Q_i = zW_i^Q$, $K_i = zW_i^K$, $V_i = zW_i^V$. Dropout of 0.1 is applied within each layer.

**Classification.** The final classification token is layer-normalised and projected to the label
space:

$$\hat{y} = \text{softmax}\big(W_{\text{head}}\,\text{LN}(z_L^{0}) + b\big), \qquad
W_{\text{head}} \in \mathbb{R}^{256 \times 512}$$

**Design rationale.** Standard ViT tokenises by linear projection of raw 16×16 pixel patches, which
supplies the encoder with no spatial prior whatsoever and is exactly why ViT needs web-scale
pretraining to compete [22]. Substituting a pretrained convolutional stack for that projection means
each token already encodes a locally translation-equivariant, hierarchically composed feature —
ImageNet's inductive bias arrives with the weights. Attention then operates over 50 tokens rather
than the 197 of ViT-B/16, an approximately 15× reduction in attention cost, which is what makes
CPU-only inference tractable (Section 3.6). For compound characters specifically, the motivation is
that a conjunct is defined by the *relationship* between its constituent consonant forms and their
placement relative to the head-line; convolution captures the constituents, and self-attention over
the full glyph captures the relationship.

**Figure 3** (`fig3_architecture.png`) shows the data flow with tensor shapes annotated at each
stage.

**Table 2. ExtendedViT layer summary.**

| Stage | Output shape | Params |
|---|---|---:|
| Input | 3 × 224 × 224 | — |
| ResNet18 conv1 + bn1 + maxpool | 64 × 56 × 56 | 9,536 |
| ResNet18 layer1–layer4 | 512 × 7 × 7 | 11,166,912 |
| Flatten → tokens | 49 × 512 | — |
| CLS token + positional embedding | 50 × 512 | 26,112 |
| Transformer encoder × 4 | 50 × 512 | 12,608,000 |
| LayerNorm + linear head | 256 | 132,352 |
| **Total** | | **23,942,912** |

### 3.5 Baseline Architectures

Three baselines span the categories identified in the review.

**Scratch CNN.** A compact separable-convolution network — one standard 3×3 convolution followed by
three depthwise-separable blocks with max-pooling, global average pooling, and a 128-unit hidden
layer — totalling 62,208 parameters, trained from random initialisation. It quantifies what is
achievable without any transfer learning.

**ResNet18 (ImageNet).** The same backbone ExtendedViT uses, with a linear classifier replacing the
1000-way head, fine-tuned end to end. It is the control that isolates the contribution of the
transformer encoder: any difference between it and ExtendedViT is attributable to the attention
stack, since the convolutional trunk is identical.

**ViT-B/16 (ImageNet).** A standard Vision Transformer with 16×16 patch embedding and 12 encoder
layers (86.3M parameters), fine-tuned with a custom head. It represents the pure-transformer
approach that [17] found superior on basic characters.

### 3.6 Training Configuration and Inference

**Optimisation.** All models use AdamW with weight decay 10⁻⁴, cross-entropy loss with label
smoothing 0.1, and a cosine-annealing schedule decaying to 10⁻⁶ over 8 epochs at batch size 32.
Gradient norms are clipped to 1.0. Pretrained models use discriminative learning rates: the
pretrained backbone trains at one tenth the rate of newly initialised parameters, preventing the
randomly initialised head from destroying pretrained features early in training. Base learning rates
are 10⁻³ for the scratch CNN, 3×10⁻⁴ for ResNet18 and ExtendedViT, and 10⁻⁴ for ViT-B/16, which
diverges at higher rates. Model selection is by best validation accuracy.

**CPU inference.** Deployment targets are assumed to lack a GPU, so inference latency is measured
single-image on CPU. This is a design constraint rather than an afterthought: it is what motivates
the 49-token design over full patch tokenisation, and it is reported alongside accuracy in Section 6
so that the accuracy/latency trade-off is explicit.

**Implementation.** PyTorch 2.13 with torchvision 0.28 and timm 1.0.28, trained on Apple Silicon via
the Metal Performance Shaders backend.

---

## 4. Experimental Design

Two data budgets are evaluated.

**Full corpus.** All 681,309 images, 545,289 of them for training. This is the headline setting and
reflects the best attainable performance.

**Equal budget.** At most 300 images per class sampled before splitting, yielding 56,414 training,
7,035 validation, and 7,035 test images. This exists because ViT-B/16 requires approximately 70 hours
per full-corpus run on our hardware, against 3 to 13 hours for the other three architectures.
Rather than compare a data-starved ViT against fully trained rivals — which would measure the data
budget rather than the architecture — we equalise the budget so all four are directly comparable.

**Controlled budget sweep.** Neither setting above isolates the effect of data volume, for two
reasons. First, they differ in the mix of source corpora as well as in size. Second, and more
seriously, both pretrained baselines silently inherited previously trained Bengali weights —
ResNet18 from an earlier RAS-only checkpoint, ExtendedViT from the full-corpus ResNet18 — so at a
restricted budget each had seen data the comparison nominally withheld. Removing that inheritance
changes first-epoch validation accuracy from 0.703 to 0.234, so its effect is large.

We therefore run a third experiment designed specifically to measure data efficiency. Training is
restricted to **Ekush alone** (122 classes, one image convention, per-class counts near-uniform at
3,056–3,079), with labels remapped to a contiguous 122-class space and per-class budgets of 50, 100,
300 and 1,000. Both architectures are initialised from ImageNet only, so neither inherits
Bengali-pretrained weights. Per-class budget is thus the sole variable. The 50 and 100 settings are
repeated across three seeds, varying initialisation, shuffling and augmentation while holding the
split fixed, so all runs share one test set and the variance measured is training stochasticity.

**Metrics.** We report top-1 accuracy; character error rate, which for isolated single-character
classification equals the top-1 error rate; macro-averaged F1; negative log-likelihood; expected
calibration error over 15 equal-width confidence bins; mean predicted confidence; parameter count;
checkpoint size; and single-image CPU latency. Accuracy is additionally disaggregated by source
corpus. ECE is included because cross-entropy conflates calibration with accuracy, and the effect we
find is specifically one of calibration.

---

## 5. Results

### 5.1 Full Corpus

**Table 3. Full corpus (681,309 images, 256 classes; 68,010 test images).**

| Method | Params | Test acc | CER | Macro-F1 | RAS | Ekush | MatriVasha |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch CNN | 0.06M | 0.6786 | 0.3214 | 0.6015 | 0.0129 | 0.7070 | 0.6614 |
| ResNet18 (ImageNet) | 11.31M | **0.9764** | 0.0236 | 0.9779 | 0.9910 | 0.9748 | 0.9779 |
| ExtendedViT (ours) | 23.94M | 0.9760 | 0.0240 | **0.9780** | 0.9936 | 0.9743 | 0.9777 |
| ViT-B/16 (ImageNet) | 86.33M | not run (≈70 h) | — | — | — | — | — |

ResNet18 and ExtendedViT are separated by 0.04 accuracy points — 27 images out of 68,010 — and their
macro-F1 scores differ by 0.0001 in the opposite direction. The two are indistinguishable when data
is abundant. Note that ResNet18's figure is, if anything, favourable to it: this run inherited an
earlier RAS-only Bengali checkpoint (Section 4).

The scratch CNN result is instructive beyond its aggregate 0.6786. Its accuracy on RAS-Compound is
**0.0129** against 0.7070 on Ekush. RAS-Compound contributes 1.1% of the images but 27 classes found
in no other corpus; without a pretrained prior the model effectively never learns them, while the
pretrained models reach 0.99 on the same classes. On a merged corpus of unequal sources, transfer
learning is not an optimisation — it is what makes the minority source learnable at all.

### 5.2 Equal Budget (300 images per class)

**Table 4. Equal budget (70,484 images, 256 classes, 300 max per class; 7,035 test images).**

| Method | Params | Test acc | CER | Macro-F1 | Backbone init |
|---|---:|---:|---:|---:|---|
| Scratch CNN | 0.06M | 0.2115 | 0.7885 | 0.1934 | random |
| ResNet18 (ImageNet) | 11.31M | 0.9544 | 0.0456 | 0.9577 | ⚠ RAS-only checkpoint |
| ExtendedViT (ours) | 23.94M | 0.9576 | 0.0424 | 0.9609 | ImageNet |
| **ViT-B/16 (ImageNet)** | 86.33M | **0.9662** | **0.0338** | **0.9689** | ImageNet |
| *ExtendedViT, warm-started* | *23.94M* | *0.9719* | *0.0281* | *0.9739* | *⚠ full-corpus checkpoint* |

The final row is reported for transparency and **excluded from comparison**: its backbone was
initialised from a ResNet18 trained on the entire corpus, so it had access to data this setting
withholds. Its 0.9719 is not a valid equal-budget result, and the 1.75-point margin over ResNet18 it
appears to show is an artifact of that inheritance.

Among the properly initialised models, ViT-B/16 is strongest at 0.9662, ahead of ExtendedViT's
0.9576 and ResNet18's 0.9544 — and ResNet18's figure is itself inflated by an inherited RAS-only
checkpoint. That pure ViT leads here is consistent with Parvez et al. [17], who found ViT superior
to VGG-16 and ResNet-50 on Bengali basic characters; we reproduce that ordering on compound
characters and a 256-class label space. It comes at 3.6× the parameters and 3.3× the inference cost
(Section 5.6).

**Figure 7** (`fig7_results.png`) summarises Tables 3 and 4, with the warm-started run shown hatched
and marked excluded.

The scratch CNN's 0.2115 against its 0.6786 on the full corpus — a 46-point swing from data volume
alone, versus roughly 2 points for the pretrained models — is the largest data-efficiency effect in
the study, and it belongs to the baseline rather than to any proposed architecture.

**Figure 6** (`fig6_training_curves.png`) gives validation accuracy and loss per epoch. ExtendedViT
leads from the first epoch and converges fastest; the inset resolves the three pretrained models,
which are otherwise indistinguishable against the scratch CNN's trajectory. Note that part of
ExtendedViT's first-epoch lead is attributable to the warm-start discussed in Section 6.

### 5.3 Controlled Budget Sweep: No Accuracy Advantage

Tables 3 and 4 differ in source composition as well as size, and both contain contaminated
initialisations, so neither isolates data volume. The Ekush-only sweep does. **Figure 10**
(`fig10_data_efficiency_sweep.png`) plots the outcome; Table 5 gives the numbers, with 50 and 100
per class averaged over three seeds and reported as mean ± standard deviation.

**Table 5. Ekush-only sweep (122 classes, ImageNet init throughout). Accuracy in points.**

| Images/class | ResNet18 | ExtendedViT | Difference | t |
|---:|---:|---:|---:|---:|
| 50 | 88.20 ± 0.59 | 89.89 ± 1.12 | +1.70 ± 0.73 | 2.33 |
| 100 | 92.05 ± 0.78 | 92.40 ± 0.31 | +0.35 ± 0.49 | 0.73 |
| 300 | 94.86 | 94.75 | −0.11 | — |
| 1000 | 96.54 | 96.40 | −0.14 | — |

**No accuracy advantage survives.** The largest difference, +1.70 points at 50 images per class,
carries a standard error of 0.73 (t = 2.33, p ≈ 0.10 on three seeds) and is not significant;
ExtendedViT's own seed-to-seed spread at that budget, ±1.12, exceeds the effect it would
demonstrate. At 100 the difference is +0.35 ± 0.49, and at 300 and 1,000 ResNet18 is fractionally
ahead. A single-seed run at 50 per class initially showed +2.46 points, which repetition reduced to
+1.70 — an illustration of why single-seed differences at this magnitude should not be reported as
findings.

We therefore state plainly that the hypothesis motivating this work — that a Transformer encoder
atop a convolutional tokenizer generalises more efficiently from few examples, in top-1 terms — is
**not supported**.

### 5.4 Calibration, and Why It Is Not an Architectural Advantage

The same runs separate decisively on the quality of their probability estimates.

**Table 6. Expected calibration error, Ekush-only sweep (15 bins; lower is better).**

| Images/class | ResNet18 | ExtendedViT | Ratio | t |
|---:|---:|---:|---:|---:|
| 50 | 0.245 ± 0.005 | **0.106 ± 0.010** | 2.31× | 22.4 |
| 100 | 0.168 ± 0.008 | **0.079 ± 0.002** | 2.13× | 17.7 |
| 300 | 0.123 | **0.064** | 1.91× | — |
| 1000 | 0.095 | **0.067** | 1.41× | — |

ExtendedViT approximately halves ResNet18's calibration error at every budget. The separation is
large relative to seed variance (t > 17 at both replicated budgets, with non-overlapping ranges),
and the ratio narrows monotonically as data grows — 2.31×, 2.13×, 1.91×, 1.41× — the shape expected
of an effect that compensates for limited data. Negative log-likelihood follows the same pattern
(0.498 vs 0.717 at 50 per class; 0.234 vs 0.240 at 1,000).

The mechanism is visible in mean predicted confidence. At 50 images per class ResNet18 averages
0.638 confidence against 0.887 accuracy, a 25-point under-confidence gap; ExtendedViT averages 0.794
against 0.911, a 12-point gap. Both are under-confident — label smoothing of 0.1 caps attainable
confidence, and this inflates the absolute ECE of both models — but ResNet18 markedly more so. Since
both models train with identical smoothing, the comparison is unaffected; absolute values should not
be compared against ECE figures from studies that omit smoothing.

The encoder is therefore not finding additional correct answers. It is producing sharper, better
calibrated probability estimates from the same evidence — which raises the question of whether the
architecture is required to obtain them.

**It is not.** Temperature scaling [Guo et al., 2017] is the standard post-hoc calibration
correction: divide the logits by a single scalar fitted by minimising validation NLL, leaving
accuracy exactly unchanged. Applying it to both models dissolves the gap.

**Table 7. Temperature scaling, Ekush-only sweep. Accuracy is unchanged by construction.**

| Budget | Model | T | ECE raw | ECE scaled | NLL raw | NLL scaled |
|---:|---|---:|---:|---:|---:|---:|
| 50/class | ResNet18 | 0.552 | 0.2496 | **0.0250** | 0.7174 | 0.4697 |
| 50/class | ExtendedViT | 0.749 | 0.1171 | 0.0341 | 0.4985 | 0.4225 |
| 100/class | ResNet18 | 0.628 | 0.1773 | **0.0232** | 0.4870 | 0.3338 |
| 100/class | ExtendedViT | 0.787 | 0.0818 | 0.0152 | 0.3887 | 0.3285 |

At 50 images per class a temperature-scaled ResNet18 (0.0250) is **better calibrated than an
unscaled ExtendedViT** (0.1171, and 0.0341 once itself scaled). The fitted temperatures are below 1
for both models, confirming that both were under-confident — an expected consequence of the label
smoothing used throughout — with ResNet18 simply further from calibrated than the hybrid.

The encoder was therefore compensating for a defect that one scalar removes more cheaply and more
completely. **We conclude that the calibration difference is not an architectural advantage.** It is
a difference in how far each architecture's raw confidences sit from their empirical accuracy under
label smoothing, and it is fully correctable post hoc. Reporting it as an architectural property —
as we were prepared to do before running this control — would have been wrong.

The broader methodological point generalises beyond this paper: uncorrected ECE comparisons between
architectures measure the interaction between an architecture and a training recipe, not a property
of the architecture. Any calibration comparison should report temperature-scaled figures alongside
raw ones.

### 5.5 Cross-Corpus Generalisation

A within-corpus split measures whether a model learned the characters; it does not measure whether
it learned them independently of how a particular project collected its images. Since merging
corpora implicitly promises the latter, we test it directly: train on one corpus, evaluate on
another, restricted to the classes the two share so the label space is identical on both sides. The
only thing that changes between training and test is the collection convention — resolution, ink
polarity, and writing population.

RAS-Compound and MatriVasha share 58 classes; RAS-Compound and Ekush share 37. Ekush and MatriVasha
share only 8, too few to be informative, so that pair is omitted. Backbones are ImageNet-initialised
only, since a Bengali checkpoint would leak the target corpus into the source model.

**Table 8. Cross-corpus generalisation (300 images per class, shared classes only).**

| Architecture | Train → Test | Shared | Same-corpus val | Cross-corpus test | Chance | Gap |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 | MatriVasha → RAS | 58 | 0.9678 | 0.0317 | 0.017 | 0.936 |
| ExtendedViT | MatriVasha → RAS | 58 | 0.9632 | 0.0375 | 0.017 | 0.926 |
| ResNet18 | RAS → MatriVasha | 58 | 0.9915 | 0.0268 | 0.017 | 0.965 |
| ExtendedViT | RAS → MatriVasha | 58 | 0.9887 | 0.0254 | 0.017 | 0.963 |
| ResNet18 | Ekush → RAS | 37 | 0.9658 | 0.0514 | 0.027 | 0.914 |
| ExtendedViT | Ekush → RAS | 37 | 0.9604 | 0.0547 | 0.027 | 0.906 |
| ResNet18 | RAS → Ekush | 37 | 1.0000 | 0.0418 | 0.027 | 0.958 |
| ExtendedViT | RAS → Ekush | 37 | 1.0000 | 0.0205 | 0.027 | 0.980 |

**Generalisation across corpora collapses to chance.** Every model scores 0.96–1.00 on its own
corpus's held-out split and 0.021–0.055 on another corpus's images of *the same characters* — between
0.8× and 2.0× the chance rate of a 37- or 58-way classifier. The result holds in both directions,
for both architectures, and across both corpus pairs. Figure 11 (`fig11_cross_source.png`) plots it.

**This is not a labelling artifact.** Near-chance transfer would follow trivially if the corpora's
class mappings disagreed, which would also invalidate the merged benchmark. We tested this directly:
the merged full-corpus model classifies RAS-Compound's shared-class test samples at 0.9921 and
MatriVasha's at 0.9800. Both corpora's renderings of a class map to the same label correctly, so the
grapheme-level reconciliation of Section 3.2 is sound and the collapse reflects genuine domain shift.

**What this means.** A model trained on a single Bengali corpus does not learn to recognise Bengali
compound characters; it learns to recognise one project's rendering of them — its resolution, its
binarisation, its ink polarity, its writer population. The 97.6% our merged model achieves across all
three corpora is attainable only because it saw all three in training. It is not evidence of a
general character recogniser, and neither, by extension, are the near-ceiling accuracies the
literature reports on individual corpora [11], [12], [17], [23]: those numbers describe within-corpus
competence and may substantially overstate performance on newly collected handwriting.

The finding reframes what merging is for. Merging is usually motivated as enlarging the training set;
here it is better understood as **covering domains**, since no single corpus yields a model that
transfers to another. It also identifies the concrete gap the field should target — domain-invariant
representations for handwritten Bengali — which the preprocessing of Section 3.3 evidently does not
achieve, despite normalising polarity and scale.

**Caveats.** Training was capped at 300 images per class; larger budgets might narrow the gap,
though the effect size makes it implausible that they close it. The two RAS-trained rows reach
1.0000 same-corpus validation, which suggests unusually low intra-class variation in that corpus and
inflates the gap for those directions specifically. And each pair is a single run, though the
consistency across eight runs and four directions makes seed variance an unlikely explanation for an
effect of this magnitude.

### 5.6 Inference Cost

**Figure 9** (`fig9_efficiency.png`) plots accuracy against single-image CPU latency measured on one
thread. ExtendedViT runs in 17 ms against ViT-B/16's 56 ms — 3.3× faster with 3.6× fewer parameters —
because attention operates over 50 tokens rather than 197. Relative to ResNet18's 13 ms, the
transformer encoder costs approximately 4 ms per image, which is the price of the data-efficiency
gain reported above. For the resource-constrained deployment scenarios the literature emphasises
[18], [20], this positions ExtendedViT favourably: it is the most accurate model at the equal-budget
setting while remaining within a few milliseconds of the cheapest competitive one.

### 5.7 Attention Analysis

**Figure 8** (`fig8_attention.png`) visualises the classification token's attention over the 7×7 token
grid in the final encoder layer, upsampled and overlaid on the input glyph. A consistent pattern
emerges: attention concentrates on the conjunct body and systematically avoids the মাত্রা head-line.
This is the behaviour the architecture was designed to produce. The head-line is present in nearly
every Bengali character and therefore carries almost no class-discriminative information, while the
structural relationship between conjoined consonant forms beneath it carries nearly all of it. That
the model learns to allocate attention accordingly — without any supervision directing it to do so —
supports the argument in Section 3.4 that global self-attention over the whole glyph is the right
inductive addition for this problem.

Attention weights are extracted by running the encoder stack manually rather than through
`nn.TransformerEncoder`; the manual pass is asserted to reproduce the model's own logits, which also
serves as a check on the formulation given in Section 3.4.

### 5.8 Ablation Studies

**Backbone initialisation.** Holding everything else fixed, initialising ExtendedViT's trunk from
ImageNet rather than from the full-corpus ResNet18 checkpoint costs 1.43 accuracy points at the
equal-budget setting (0.9719 → 0.9576). The inherited trunk had seen the entire corpus, so that
1.43-point difference measures leaked information rather than architecture. The equivalent
inheritance in ResNet18 — from a RAS-only checkpoint — moves first-epoch validation accuracy from
0.703 to 0.234 in the single-source sweep. Both baselines were affected; Sections 5.3 and 5.4 use
ImageNet initialisation throughout.

**Temperature scaling.** Run, and reported in Section 5.4: a single scalar fitted on validation data
removes the calibration difference between architectures entirely. This was the decisive control for
the paper's remaining architectural claim, and it did not survive it.

**Ablations worth adding if reviewers ask, or if space permits.** Each isolates one design decision
and costs roughly 1.5 hours at the equal-budget setting. Given that no architectural advantage
survived, these are now of limited value; the `L = 0` variant is the only one we would prioritise,
since it reduces exactly to the ResNet18 baseline and would confirm the encoder's contribution is
null rather than merely small.

| Ablation | Question it answers | Variants |
|---|---|---|
| Encoder depth | Is 4 layers the right capacity, or is the gain saturating? | L = 0 (≡ ResNet18 + pooling), 2, 4, 8 |
| Attention heads | Does head count matter at 50 tokens? | h = 4, 8, 16 |
| Positional embeddings | Do the 49 tokens need explicit position, given convolution already encodes it? | learned / none |
| Polarity normalisation | How much does the preprocessing contribute versus the architecture? | on / off |
| Backbone depth | Would a shallower trunk with a deeper encoder do better? | ResNet18 layer3 output (14×14, 196 tokens) vs layer4 (7×7, 49) |

The `L = 0` variant is the most informative single ablation, since it reduces exactly to the
ResNet18 baseline and isolates the encoder's entire contribution.

### 5.9 Comparison with Prior Published Results

**Table 9. Positioning against prior Bengali character recognition work.**

| Study | Architecture | Dataset | Classes | Reported |
|---|---|---|---:|---:|
| Das et al. (2014) [3] | Convex-hull + quad-tree + SVM | CMATERdb 3.1.3.3 | 171 | 79.35% |
| Roy et al. (2017) [9] | Layer-wise DCNN | CMATERdb 3.1.3.3 | 171 | 90.33% |
| Chatterjee et al. (2019) [11] | ResNet-50, ImageNet | BanglaLekha-Isolated | 84 | 96.12% |
| Ghosh et al. (2020) [12] | InceptionResNetV2 / DenseNet121 | CMATERdb | 231 | 96.99% |
| CompoundDenseNet [23] | Modified DenseNet | BanglaLekha / Ekush / CMATERdb | varies | 96.2–98.5% |
| Parvez et al. (2025) [17] | ViT-B/16 | CMATERdb 3.1.2 (basic) | 50 | 98.26% |
| ResViT [19] | ResNet50v2 + ViT, parallel fusion | BanglaLekha-Isolated | 84 | 97.21% |
| Raquib et al. [25] | EfficientNetB3 + ViT + Conformer | Own (78-class) / CHBCR | 78 | 98.84% / 96.49% |
| **This work** | **ResNet18 tokenizer + Transformer encoder** | **Unified RAS + Ekush + MatriVasha** | **256** | **97.56%** (full corpus) |

**These numbers are not a leaderboard and must not be presented as one.** Class counts range from 50
to 256, evaluation protocols and splits differ, several sets are near-balanced while ours is
deliberately not, and some reported figures are averages over balanced subsets. A 98.26% on 50 basic
characters [17] is a substantially easier problem than 97.56% on 256 classes including the full
compound inventory and a 27-class rare tail. The table's purpose is to show that our result sits in
the range established by recent deep architectures while being obtained on a markedly larger and
more imbalanced label space — not to claim state of the art. Any venue submission should state this
caveat explicitly rather than letting the table imply a ranking.

### 5.10 Per-Source and Per-Class Analysis

Per-source accuracy is reported in Tables 3 and 4. All three pretrained models score highest on
RAS-Compound (0.991–0.994) and lowest on Ekush (0.974), the reverse of what corpus size alone would
predict — RAS-Compound is the smallest source but has the cleanest, highest-resolution binarised
glyphs, while Ekush is natively 28×28. Image quality dominates corpus size here. The scratch CNN
inverts this completely (0.013 on RAS-Compound, 0.707 on Ekush), because without a pretrained prior
it learns only what it sees in quantity.

**Rare classes are the easiest, not the hardest.** The 27 classes with fewer than 100 images reach a
mean per-class F1 of **0.9966** (ExtendedViT; 0.9925 for ResNet18) against **0.9758** for the 229
well-populated classes — and all 27 achieve F1 = 1.0. This inverts the usual expectation and follows
directly from the confound established in Section 3.2: those 27 classes are contributed solely by
RAS-Compound, whose images are high-resolution and cleanly binarised, and no other corpus supplies a
competing rendering of them. Their scarcity is an artifact of source size, while their separability
comes from image quality. On this benchmark, therefore, **macro-F1 does not measure long-tail
performance**, and neither macro-F1 nor per-class accuracy should be read as evidence about rare
conjuncts. We report macro-F1 as a descriptive statistic only.

**Errors concentrate on minimally different conjunct pairs.** Only 7 of 256 classes fall below F1
0.90. The most-confused pairs are near-identical, and both architectures fail on essentially the
same ones:

**Table 10. Most-confused class pairs, full corpus (ExtendedViT / ResNet18 error counts).**

| True → Predicted | ExtendedViT | ResNet18 | What differs |
|---|---:|---:|---|
| ন্ড্র → ণ্ড্র | 33 | 25 | dental ন vs retroflex ণ |
| ণ্ড্র → ন্ড্র | 24 | 28 | the same pair, reversed |
| ম্ভ্র → ন্ত্র | 32 | 26 | both components |
| ত্ত → ও | 25 | 26 | conjunct resembling a vowel |
| ম্ন → ম্ম | 19 | 19 | ন vs ম as second component |
| ন্ঢ → ন্ট | 15 | 16 | ঢ vs ট |
| শ্ন → শ্ম | 12 | 13 | ন vs ম as second component |
| ন্ব → ণ্ব | 12 | 12 | dental ন vs retroflex ণ |
| ত্থ → থ | 12 | — | conjunct vs its own second component |
| ২ → হ | 13 | 13 | digit against a consonant |

Three systematic patterns emerge. First, the **dental/retroflex nasal contrast** (ন vs ণ) accounts
for the single largest error source, and it is symmetric — the confusion runs in both directions,
indicating genuine visual ambiguity rather than a class prior. Second, **ন vs ম as a second
component** recurs across otherwise unrelated conjuncts. Third, conjuncts are confused with **their
own constituents** (ত্থ → থ) or with visually similar non-conjuncts (ত্ত → ও, ২ → হ), which is a
failure of compositional structure rather than of stroke detection.

That both architectures fail on the same pairs, in similar proportions, indicates these errors are
properties of the data rather than of the model — consistent with the near-identical aggregate
accuracies of Table 3, and with the inter-class similarity the literature identifies as the field's
persistent obstacle [17], [23], [32]. Full per-class results are in Appendix C; confusion matrices
are `results/confusion_full_*.png`.

**Per-source accuracy inverts corpus size.**
corpus to establish whether the merge benefits all three; report macro-F1 against accuracy to
quantify performance on the 27 rare classes; include confusion matrices for the two leading models,
focusing on the structurally similar conjunct pairs that the literature identifies as the persistent
failure mode [17], [23].

---

## 6. Discussion

**The dominant effect is the corpus, not the architecture.** Section 5.5 puts the architecture
comparison in proportion. Every difference between architectures reported in this paper is at most
a few accuracy points; the difference between evaluating within a corpus and across corpora is
ninety. Any conclusion about which architecture is best for Bengali compound characters is therefore
conditional on a fixed collection convention, and would need re-establishing on data collected
differently. We state this before discussing the architecture results, because it bounds how much
weight they can carry.

**What the architecture results do not support.** We set out expecting the hybrid to be more accurate than a
convolutional baseline when data is limited. It is not. Across a controlled sweep with the sole
variable being per-class budget, and with three seeds at the two smallest budgets, no accuracy
difference reaches significance; at the two largest budgets ResNet18 is fractionally ahead. We
report this because the alternative — presenting the single-seed +2.46 points at 50 images per
class, or the 1.75-point margin from the contaminated warm-started run — would have been a finding
that repetition and a proper control both dissolve.

**What the results do support.** The encoder substantially improves the calibration of the model's
probability estimates: roughly a halving of expected calibration error at every budget, with the
advantage largest when data is scarcest and narrowing monotonically as data grows. This is a
genuine architectural effect, not a data artifact — the sweep holds corpus, image convention,
class inventory, initialisation, optimiser and schedule fixed, and the separation is an order of
magnitude larger than seed variance.

We suggest the following interpretation. Self-attention over the whole glyph aggregates evidence
from spatially distant strokes before a decision is made, whereas a convolutional hierarchy commits
to increasingly abstract local features and pools them only at the end. When training data is
plentiful, both routes reach the same decision boundary and top-1 accuracy converges. When data is
scarce, the convolutional model's uncertainty is poorly estimated — it is markedly under-confident
(0.638 mean confidence at 0.887 accuracy) — while the attention-based aggregation yields
probabilities much closer to observed frequencies. The attention maps of Section 5.7, which show
the classification token consistently ignoring the shared মাত্রা head-line and concentrating on the
discriminative conjunct body, are consistent with this account, though they do not establish it.

**Why calibration matters here.** Isolated-character accuracy is rarely the end goal. In word- and
line-level recognition the character posterior feeds a decoder — beam search, a language model, or
a lexicon-constrained search — which consumes probabilities rather than argmax labels. A recogniser
whose confidences are twice as well calibrated supplies a materially better signal to that decoder
even when its top-1 accuracy is identical. For Bengali specifically, where conjuncts are mutually
confusable and a decoder must arbitrate between close alternatives, the shape of the posterior is
arguably more consequential than its mode.

**Relation to prior work.** That ViT-B/16 leads the properly initialised equal-budget comparison
(Table 4) reproduces, on compound characters and a 256-class label space, the ordering Parvez et al.
[17] reported for basic characters. Our contribution relative to that finding is not to overturn it
but to quantify its cost: ViT-B/16 buys roughly one accuracy point over ExtendedViT at 3.6× the
parameters and 3.3× the inference time. Numbers are not directly comparable across studies given
differing class counts and protocols, but for orientation ResViT reports 97.21% on
BanglaLekha-Isolated [19] and CompoundDenseNet 96.2–98.5% across three benchmarks [23], both on
smaller class inventories than the 256 used here.

**Threats to validity.**

*Initialisation contamination, now controlled.* Both pretrained baselines initially inherited
previously trained Bengali weights — ResNet18 from a RAS-only checkpoint, ExtendedViT from the
full-corpus ResNet18 — which invalidated our first equal-budget comparison. Removing that
inheritance moves first-epoch validation accuracy from 0.703 to 0.234. Every number in Sections 5.3
and 5.4 comes from runs initialised from ImageNet alone. Tables 3 and 4 retain the inherited
initialisations and are annotated accordingly; they should be read as upper bounds on the
baselines rather than as controlled comparisons.

*Seeds and statistical power.* Three seeds establish that the accuracy differences are not
significant and that the calibration differences are, but three is a small sample; the 300 and 1,000
per-class budgets are single runs, so their calibration figures carry no error estimate.

*Calibration measurement.* Label smoothing of 0.1 depresses attainable confidence and inflates the
absolute ECE of both models. The comparison is unaffected because smoothing is identical, but our
absolute ECE values should not be compared against studies that omit it, and a replication without
smoothing would strengthen the claim.

*Epoch budget.* All models train for 8 epochs, which suits fine-tuning but may underserve the
scratch CNN, whose 0.2115 should be read as its 8-epoch result at that budget rather than a ceiling.

**Limitations.** Evaluation is on isolated characters; performance within connected words, where
segmentation error compounds classification error, is not measured — and it is precisely there that
the calibration advantage would have to be demonstrated to matter practically. The sweep uses one
corpus, so whether the calibration effect generalises across image conventions is untested. The
corpora may not represent natural document imagery. And because class rarity is perfectly confounded
with source identity (Section 3.2), we make no claim about rare-class behaviour.

---

## 7. Conclusion

We unified three Bengali handwriting corpora into a 256-class, 681,309-image benchmark by
reconciling labels at the level of Unicode-normalised graphemes, and used it to evaluate four
architectures under an identical protocol.

The benchmark's principal finding is negative and, we believe, consequential for the field. Trained
on any one constituent corpus and tested on another — restricted to the classes they share, so the
label space is identical on both sides — every architecture falls from 0.96–1.00 same-corpus accuracy
to 0.02–0.05 cross-corpus, at or barely above chance. We verified that this reflects domain shift
rather than label disagreement, since the merged model classifies both corpora's renderings of a
shared class correctly (0.9921 and 0.9800). A model trained on one Bengali corpus therefore does not
learn to recognise Bengali compound characters; it learns to recognise that project's rendering of
them. Single-corpus accuracies — including the near-ceiling figures widely reported in this
literature — describe within-corpus competence and should not be read as evidence of generalisation.
Merging corpora is best understood not as enlarging a training set but as covering domains.

Against that backdrop, architecture differences are small. Four architectures spanning 0.06M to
86.3M parameters differ by at most about one accuracy point once initialisation is controlled.
ExtendedViT — an ImageNet-pretrained ResNet18 acting as a learned tokenizer for a four-layer
Transformer encoder — matches a fine-tuned ResNet18 on the full corpus (0.9760 vs 0.9764) and shows
**no significant accuracy advantage at any budget** in a controlled sweep with three seeds. An
apparent halving of the baseline's expected calibration error proves fully correctable by
temperature scaling, which leaves the scaled baseline better calibrated than the unscaled hybrid; it
is therefore a property of the training recipe rather than of the architecture. ViT-B/16 is the most
accurate model at an equal budget, at 3.6× the parameters and 3.3× the inference cost of the hybrid.

Three methodological cautions follow, each discovered by attempting to falsify a claim of our own.
Warm-starting from a previously trained checkpoint — routine in this literature — inflated our
equal-budget result by 1.43 accuracy points. Macro-F1 does not measure long-tail performance on this
corpus: the 27 rarest classes score a *higher* mean F1 (0.9966) than the 229 common ones (0.9758),
because rarity is confounded with source image quality. And architecture-level calibration
comparisons should always be reported temperature-scaled, since uncorrected ECE measures the
interaction of architecture and training recipe rather than a property of the architecture. We also
note that without pretraining a scratch CNN reaches 0.0129 accuracy on the minority corpus supplying
27 otherwise-unavailable classes, against 0.707 on the largest corpus — on a merged corpus of
unequal sources, transfer learning is what makes the minority source learnable at all.

Future work follows directly from the cross-corpus result. The field needs domain-invariant
representations for handwritten Bengali: our preprocessing normalises polarity and scale and is
plainly insufficient, so stronger domain generalisation — adversarial or style-invariant training,
heavy style augmentation, or test-time adaptation — is the obvious next target, and cross-corpus
accuracy is the metric that should report progress on it. We would also encourage the field to
adopt held-out-corpus evaluation as standard practice alongside within-corpus splits, since the two
measure very different things.

On the architecture side, our results suggest the returns from further model design on isolated
Bengali compound characters are limited, and that effort is better directed at the evaluation and
data problems above. Where architecture does matter is cost: establishing the accuracy-per-millisecond
frontier for CPU-only deployment, which our four points sketch but do not map.

---

## 8. Reproducibility

All results are produced by a single codebase in which dataset scanning, label reconciliation,
splitting, and image transformation are shared across architectures; only the model varies. The
train/validation/test partition is derived deterministically from a fixed seed (42) applied to a
per-class shuffle, so the partition is identical for every run reported here and can be regenerated
exactly.

**Table 11. Complete training configuration.**

| Setting | Value |
|---|---|
| Input resolution | 224 × 224, 3-channel grayscale |
| Batch size | 32 |
| Epochs | 8 |
| Optimiser | AdamW, weight decay 10⁻⁴ |
| Base learning rate | 10⁻³ (CNN), 3×10⁻⁴ (ResNet18, ExtendedViT), 10⁻⁴ (ViT-B/16) |
| Backbone learning rate | 0.1 × base for all pretrained models |
| Schedule | Cosine annealing to 10⁻⁶ over 8 epochs |
| Loss | Cross-entropy, label smoothing 0.1 |
| Gradient clipping | Max norm 1.0 |
| Augmentation | Rotation ±15°, translation ≤10%, shear 5°, brightness/contrast jitter 0.3 |
| Normalisation | μ = σ = 0.5 per channel |
| Model selection | Best validation accuracy |
| Split | 80/10/10 stratified per class, seed 42 |
| Framework | PyTorch 2.13, torchvision 0.28, timm 1.0.28 |
| Hardware | Apple Silicon, Metal Performance Shaders backend |
| Inference measurement | CPU, single image, one thread |

Commands to reproduce every result:

```bash
python train_arch.py --arch {cnn,resnet18,vit,extended_vit} [--limit-per-class 300] [--no-warm-start]
python evaluate.py [--limit-per-class 300] [--suffix _imagenet] --confusion --per-class
python cross_source.py --train <corpus> --test <corpus> --arch <arch>
python temperature_scaling.py --source ekush --limit-per-class 50
python paper/figures.py
```

---

## 9. Declarations

> Every item below is required by most journals. Nothing here can be inferred from the code or
> results — fill each one before submission.

**Data availability.** The three source corpora are pre-existing public datasets: RAS-Compound,
Ekush [Rabby et al., 2018], and MatriVasha. `[FILL: repository URL, version or access date, and
licence for each]`. The label-reconciliation mapping files (`ras_class_mapping.csv`,
`matrivasha_mapping.csv`) and the manifest-construction code that reproduces the unified 256-class
corpus are released with this paper; `[FILL: decide whether the merged corpus itself is redistributed,
which depends on whether all three source licences permit it — verify before promising a download]`.

**Code availability.** `[FILL: repository URL and DOI, e.g. an archived Zenodo release]`. Trained
checkpoints `[FILL: released / available on request]`.

**Ethics approval.** The source corpora were collected from human writers — Ekush alone from 3,086
contributors, and MatriVasha records per-writer gender metadata. This work performs secondary
analysis of already-published, de-identified datasets and collected no new human data.
`[FILL: confirm each source's original consent and ethics provenance, and state whether your
institution required review for secondary use; some venues ask explicitly]`.

**Consent to participate / publish.** Not applicable; no new human participants. `[FILL: confirm]`

**Funding.** `[FILL: grant numbers and funders, or "This research received no specific grant from
any funding agency in the public, commercial, or not-for-profit sectors."]`

**Competing interests.** `[FILL: typically "The authors declare no competing interests."]`

**Author contributions (CRediT).** `[FILL per author]` — Conceptualisation; Data curation; Formal
analysis; Investigation; Methodology; Software; Validation; Visualisation; Writing – original draft;
Writing – review & editing.

**Acknowledgements.** `[FILL: the creators of RAS-Compound, Ekush and MatriVasha for releasing their
data publicly; any colleagues, compute providers, or reviewers]`.

**Use of AI tools.** `[FILL: most venues now require disclosure of generative-AI assistance in
preparing the manuscript or code. State what was used and for what.]`

---

## Appendix A. Notation

| Symbol | Meaning |
|---|---|
| $x$ | Input image, $\mathbb{R}^{3 \times 224 \times 224}$ |
| $F$ | ResNet18 `layer4` feature map, $\mathbb{R}^{512 \times 7 \times 7}$ |
| $N = 49$ | Number of spatial tokens |
| $D = 512$ | Token dimension |
| $x_{\text{cls}}$ | Learnable classification token |
| $E_{\text{pos}}$ | Learnable positional embedding, $\mathbb{R}^{50 \times 512}$ |
| $z_\ell$ | Token sequence after encoder layer $\ell$ |
| $L = 4$ | Number of encoder layers |
| $h = 8$, $d_k = 64$ | Attention heads and per-head dimension |
| $C = 256$ | Number of classes |

## Appendix B. Class Inventory

The complete inventory is released as `class_inventory.csv` (256 rows): index, character, Unicode
code points, Unicode character names, contributing corpora, and image count. It is intended as
supplementary material rather than body text.

Classes range from a single code point (e.g. ঁ, U+0981 BENGALI SIGN CANDRABINDU) to five
(ত + ্ + র + ্ + য-type conjuncts), reflecting that a conjunct is encoded as a consonant sequence
joined by U+09CD BENGALI SIGN VIRAMA rather than as an atomic code point. This is precisely why
label reconciliation must operate on NFC-normalised grapheme strings rather than on folder indices
(Section 3.2). Of the 256 classes, 97 are attested in more than one corpus and 159 in exactly one.

## Appendix C. Per-Class Results

Per-class precision, recall, F1 and support for both leading models on the full corpus are released
as `results/per_class_extended_vit.csv` and `results/per_class_resnet18.csv`, sorted ascending by
support. The complete ranked confusion lists are `results/confusions_*.csv`.
All four regenerate with:

```bash
python evaluate.py --arch extended_vit resnet18 --per-class --confusion
```

Summary: 27 classes at F1 = 1.000 (ExtendedViT) and 25 (ResNet18); 7 and 6 classes respectively
below F1 0.90; worst classes ন্ড্র (0.8652) and ণ্ড্র (0.8814) — the two halves of the same
dental/retroflex confusion discussed in Section 5.10.

---

## References

Use the numbering already established in the standalone literature review; the citations appearing in
this draft are [2]–[5], [7], [9], [11], [12], [16], [17], [19], [22]–[25], [28], [30], [32], [33],
[37], and [41]–[42].

`[FILL: the review's reference list is not yet reproduced here — paste it, then verify every
in-text citation resolves and that Ethnologue, Rabby et al. (2018) for Ekush, and the MatriVasha
and RAS-Compound dataset papers are all present, since this draft cites them by name outside the
numbered scheme.]`

**References to add** — required by the new Section 2.7 and by Section 5.4. Verify each before use;
they are cited from standard knowledge of these literatures rather than from a checked copy:

- ⚑ Torralba, A. and Efros, A. A. *Unbiased Look at Dataset Bias.* CVPR 2011. — the "name that
  dataset" demonstration and cross-dataset generalisation degradation.
- ⚑ Recht, B., Roelofs, R., Schmidt, L. and Shamir, V. *Do ImageNet Classifiers Generalize to
  ImageNet?* ICML 2019. — accuracy drops on a replication-protocol test set.
- ⚑ Ben-David, S. et al. *A Theory of Learning from Different Domains.* Machine Learning, 2010. —
  the source-error-plus-divergence bound.
- ⚑ Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. *On Calibration of Modern Neural Networks.*
  ICML 2017. — temperature scaling and expected calibration error, used in Section 5.4.
- Optionally a recent domain-generalisation survey (e.g. Zhou et al., *Domain Generalization: A
  Survey*, TPAMI 2022) if the venue expects broader positioning.

---

## Writing still required

Distinct from the experiments below — these are authoring tasks, several of which will block
submission at most venues.

**Only you can supply these.** Author list and affiliations; ORCIDs; funding statement; competing
interests; CRediT contributions; acknowledgements; AI-use disclosure; the licence and access status
of each source corpus, and whether the merged corpus may legally be redistributed. All are marked
`[FILL]` in Sections 9 and the title page.

**Needs a decision.** Target venue, which determines length limit, reference style, whether
highlights and a graphical abstract are required, and single- vs double-column layout. The draft is
currently venue-neutral and runs long for a conference; a journal such as *Pattern Recognition
Letters* or *IEEE Access* fits it better without cutting.

**Needs writing once results land.** Section 2 is a pointer to your standalone review and must be
pasted in, with the new subsection 2.7 (drafted in place) inserted and the existing 2.7–2.10
renumbered to 2.8–2.11, plus the seventh research gap added to what becomes 2.10. The reference list
must be reproduced, the five references listed at the end of this draft added and verified, and
every in-text citation checked to resolve. The abstract's final paragraph is written against provisional numbers and needs rewriting
once the ImageNet-init and ViT results are in. Figure captions are currently inline descriptions and
should be rewritten as standalone captions — a reader should understand each figure without the body
text.

**Worth adding if space permits.** A graphical abstract (Figure 11 makes the paper's central point in one image); a
notation table is already drafted as Appendix A; the full class inventory as supplementary material.

---

## Outstanding experiments before submission

1. ~~**Pending results**~~ — complete. All runs finished; every number in the draft is measured.
2. **Run `evaluate.py`** on both budgets for test accuracy, CER, macro-F1, per-source accuracy,
   latency, and confusion matrices.
3. ~~**Warm-start replication**~~ — queued; runs automatically after the training chain via
   `train_arch.py --arch extended_vit --limit-per-class 300 --no-warm-start`, writing
   `extended_vit_merged_lpc300_imagenet.pth` and `results_lpc300_imagenet.csv`.
4. **Seed repeats** — at least three seeds for ResNet18 and ExtendedViT at the equal budget.
   Without these, Table 4's margin cannot be called significant, only observed.
5. **Section 2** — paste in the existing literature review.
6. **Figures 10–11** — confusion matrices for the two leading models cropped to the most-confused
   conjunct pairs, and per-source accuracy bars. Both need `evaluate.py`, so they are blocked on
   item 1.

### Figures already built

Regenerate any of them with `./venv/bin/python paper/figures.py`. Figures 6, 7 and 9 read the
training logs directly, so they update automatically as the remaining runs finish.

| Figure | File | Section |
|---|---|---|
| 1. Overall workflow | `fig1_workflow.png` | 3.1 |
| 2. Dataset composition | `fig2_dataset.png` | 3.2 |
| 3. ExtendedViT architecture | `fig3_architecture.png` | 3.4 |
| 4. Sample glyphs across corpora | `fig4_samples.png` | 3.2 |
| 5. Preprocessing pipeline | `fig5_preprocessing.png` | 3.3 |
| 6. Training curves | `fig6_training_curves.png` | 5.2 |
| 7. Results and data efficiency | `fig7_results.png` | 5.3 |
| 8. CLS attention maps | `fig8_attention.png` | 5.5 |
| 9. Accuracy vs CPU latency | `fig9_efficiency.png` | 5.4 |
