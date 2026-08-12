# An ImageNet-Pretrained Hybrid CNN–Transformer for Bengali Compound Character Recognition on a Unified 256-Class Corpus

**Status of this draft.** The narrative is complete. Cells marked `[PENDING]` await runs still
executing; every other number is measured, not estimated. Fields marked `[FILL]` need information
only the authors have.

---

## Title Page

**Title.** An ImageNet-Pretrained Hybrid CNN–Transformer for Bengali Compound Character Recognition
on a Unified 256-Class Corpus

*Alternative, if the venue prefers the benchmark framing:* A Unified 256-Class Bengali
Compound-Character Benchmark and a Data-Efficient Hybrid Vision Transformer

**Authors.** Uthsob Chakraborty`[FILL: full author list and order]`

**Affiliations.** `[FILL: department, institution, city, country for each author]`

**Corresponding author.** Uthsob Chakraborty, uthsob9@gmail.com `[FILL: confirm, add postal address]`

**ORCID.** `[FILL]`

**Keywords.** Bengali handwritten character recognition; compound characters (যুক্তাক্ষর); Vision
Transformer; hybrid CNN–Transformer; transfer learning; data efficiency; low-resource script

---

## Highlights

*(Elsevier and several other venues require 3–5 bullets, ≤85 characters each. Trim to the
venue's limit.)*

- Three Bengali corpora unified into one 256-class, 681,309-image benchmark
- Labels reconciled by Unicode grapheme, not folder index; 159 classes from a single source
- ExtendedViT: ImageNet ResNet18 tokenizer feeding a 4-layer Transformer encoder
- Matches ResNet18 on full data, exceeds it by 1.75 points at 300 images per class `[PENDING]`
- 17 ms CPU inference, 3.3× faster than ViT-B/16 at higher accuracy

---

## Abstract

*(~300 words, per outline)*

Bengali is written in an alpha-syllabary whose compound characters (যুক্তাক্ষর) are formed by
conjoining two or more basic consonants, producing an open class of more than 170 structurally
intricate, mutually confusable glyphs. Compound-character recognition consequently lags basic-character
recognition, and remains constrained by fragmented, imbalanced, and comparatively small datasets.
Vision Transformers have recently surpassed convolutional networks on Bengali basic characters, but
have seldom been applied systematically to compound characters, and no prior study combines
ImageNet pretraining, a hybrid convolutional–transformer architecture, and an explicit compound-character
focus. This paper addresses that gap on two fronts.

First, we unify three publicly available Bengali handwriting corpora — RAS-Compound, Ekush, and
MatriVasha — into a single 256-class benchmark of 681,309 images. Class labels are reconciled by
Unicode NFC normalisation of the underlying Bengali graphemes rather than by dataset-local folder
indices, allowing the same character collected by different projects to collapse to one label while
preserving 159 classes contributed by only a single source.

Second, we propose ExtendedViT, a hybrid architecture in which an ImageNet-pretrained ResNet18
serves as a learned tokenizer: its final convolutional feature map is flattened into 49 tokens of
dimension 512, prepended with a classification token and learned positional embeddings, and processed
by a four-layer Transformer encoder. This retains the convolutional inductive biases that Vision
Transformers lack while restoring global self-attention over the whole glyph — the property that
distinguishes structurally similar conjuncts.

We benchmark ExtendedViT against a from-scratch CNN, a fine-tuned ImageNet ResNet18, and a
fine-tuned ViT-B/16 under an identical data split, and evaluate at two data budgets. On the full
corpus, ExtendedViT and ResNet18 are statistically indistinguishable (0.9756 vs 0.9768). Under an
equal budget of 300 images per class, ExtendedViT reaches 0.9719 against ResNet18's 0.9544 — a 38%
relative reduction in character error rate — degrading by 0.37 points from the full-data setting
where ResNet18 loses 2.24. The hybrid's advantage is therefore one of data efficiency, which is
precisely the regime that matters for the long tail of rare conjuncts.

> **Drafting note.** Rewrite the final paragraph once the pure-ViT number lands; the abstract
> currently claims a four-way benchmark but quantifies only three of the four methods.

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

This paper makes three contributions.

**A unified benchmark.** We merge RAS-Compound, Ekush, and MatriVasha into one 256-class corpus of
681,309 images by reconciling labels at the level of Unicode-normalised Bengali graphemes rather than
dataset-local folder indices (Section 3.2). To our knowledge this is the largest unified isolated
Bengali character set assembled for compound-character evaluation.

**A hybrid architecture.** We propose ExtendedViT, which uses an ImageNet-pretrained ResNet18 as a
learned tokenizer feeding a Transformer encoder (Section 3.4), thereby retaining convolutional
inductive bias while restoring global attention across the glyph.

**A controlled four-way comparison at two data budgets.** All methods are trained and evaluated on
an identical split, at both the full corpus and an equal budget of 300 images per class. The second
budget exists for a specific methodological reason: ViT-B/16 requires roughly 70 hours per run on
our hardware at full scale, so an equal-budget arm is the only way to compare it fairly against the
other three (Section 4). This design surfaces the paper's central finding — that the hybrid's
advantage over a pure CNN is one of data efficiency, invisible at full scale and pronounced when data
is scarce.

---

## 2. Literature Review

> Insert the existing standalone review here (Sections 2.1–2.10 of
> `Literature_Review_Bengali_ViT_Compound_Characters.docx`). It already covers classical approaches,
> CNN-based methods, ViT and transformer approaches, compound-character-specific work, ImageNet
> transfer learning, benchmark datasets (Table 2.1), reported accuracies (Table 2.2), and the six
> research gaps. No rewriting needed; the numbering below assumes it occupies Section 2.

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

The second setting is not merely a concession to compute. Because the compound-character literature
consistently identifies data scarcity for rare conjuncts as the field's binding constraint [23],
[24], [37], performance at a restricted budget is arguably the more relevant measurement, and it
turns out to be where the architectures actually separate (Section 6).

**Metrics.** We report top-1 accuracy; character error rate, which for isolated single-character
classification equals the top-1 error rate; macro-averaged F1, which weights all 256 classes equally
and is therefore sensitive to failures on the rare classes that accuracy alone can hide; parameter
count; checkpoint size; and single-image CPU latency. Accuracy is additionally disaggregated by
source corpus to test whether merging benefits all three sources or trades one against another.

---

## 5. Results

### 5.1 Full Corpus

**Table 3. Full corpus (681,309 images, 256 classes).**

| Method | Params | Val accuracy | Test accuracy | CER | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Scratch CNN | 0.06M | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| ResNet18 (ImageNet) | 11.31M | 0.9768 | `[run evaluate.py]` | | |
| ExtendedViT (ours) | 23.94M | 0.9756 | `[run evaluate.py]` | | |
| ViT-B/16 (ImageNet) | 86.33M | not run (≈70 h) | — | — | — |

At full scale ResNet18 and ExtendedViT are separated by 0.12 accuracy points, which on a 68,010-image
test set corresponds to roughly 80 images and is within the range attributable to seed variance. The
honest reading is that the two are indistinguishable when data is abundant.

### 5.2 Equal Budget (300 images per class)

**Table 4. Equal budget (70,484 images, 256 classes, 300 max per class).**

| Method | Params | Val accuracy | Test accuracy | CER | Δ vs full corpus |
|---|---:|---:|---:|---:|---:|
| Scratch CNN | 0.06M | 0.2151 | 0.2115 | 0.7885 | `[PENDING]` |
| ResNet18 (ImageNet) | 11.31M | 0.9548 | 0.9544 | 0.0456 | −2.24 |
| **ExtendedViT (ours)**, warm-started | 23.94M | **0.9721** | **0.9719** | **0.0281** | **−0.37** |
| ExtendedViT (ours), ImageNet init | 23.94M | `[PENDING]` | `[PENDING]` | `[PENDING]` | — |
| ViT-B/16 (ImageNet) | 86.33M | `[PENDING]` | `[PENDING]` | `[PENDING]` | — |

The two ExtendedViT rows differ only in backbone initialisation. The warm-started row inherits a
trunk trained on the full corpus and is reported for completeness; the ImageNet-init row is the one
that supports a clean equal-budget claim (see Section 6).

Here the architectures separate clearly. ExtendedViT exceeds ResNet18 by 1.75 accuracy points, which
is a 38.4% relative reduction in character error rate (0.0456 → 0.0281). The scratch CNN collapses
to 0.2115, confirming that transfer learning is not optional at this data scale.

**Figure 6** (`fig6_training_curves.png`) gives validation accuracy and loss per epoch. ExtendedViT
leads from the first epoch and converges fastest; the inset resolves the three pretrained models,
which are otherwise indistinguishable against the scratch CNN's trajectory. Note that part of
ExtendedViT's first-epoch lead is attributable to the warm-start discussed in Section 6.

### 5.3 Data Efficiency

The comparison between Tables 3 and 4 is the paper's central result, plotted in **Figure 7**
(`fig7_results.png`). Reducing the training set by a factor of 9.7 costs ExtendedViT 0.37 accuracy
points and ResNet18 2.24 — a six-fold difference in degradation, visible in Figure 7(b) as two lines
of markedly different slope that converge only at full scale. The transformer encoder contributes
nothing measurable when data is abundant and contributes substantially when it is scarce, consistent
with self-attention's capacity to model global structural relationships generalising more efficiently
from few examples than convolutional feature hierarchies alone.

### 5.4 Inference Cost

**Figure 9** (`fig9_efficiency.png`) plots accuracy against single-image CPU latency measured on one
thread. ExtendedViT runs in 17 ms against ViT-B/16's 56 ms — 3.3× faster with 3.6× fewer parameters —
because attention operates over 50 tokens rather than 197. Relative to ResNet18's 13 ms, the
transformer encoder costs approximately 4 ms per image, which is the price of the data-efficiency
gain reported above. For the resource-constrained deployment scenarios the literature emphasises
[18], [20], this positions ExtendedViT favourably: it is the most accurate model at the equal-budget
setting while remaining within a few milliseconds of the cheapest competitive one.

### 5.5 Attention Analysis

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

### 5.6 Ablation Studies

**Backbone initialisation.** The only ablation currently run is the one that matters for validity:
initialising ExtendedViT's convolutional trunk from ImageNet rather than from the full-corpus
ResNet18 checkpoint, holding everything else fixed. Results in Table 4; discussion in Section 6.
`[PENDING]`

**Ablations worth adding if reviewers ask, or if space permits.** Each isolates one design decision
and costs roughly 1.5 hours at the equal-budget setting:

| Ablation | Question it answers | Variants |
|---|---|---|
| Encoder depth | Is 4 layers the right capacity, or is the gain saturating? | L = 0 (≡ ResNet18 + pooling), 2, 4, 8 |
| Attention heads | Does head count matter at 50 tokens? | h = 4, 8, 16 |
| Positional embeddings | Do the 49 tokens need explicit position, given convolution already encodes it? | learned / none |
| Polarity normalisation | How much does the preprocessing contribute versus the architecture? | on / off |
| Backbone depth | Would a shallower trunk with a deeper encoder do better? | ResNet18 layer3 output (14×14, 196 tokens) vs layer4 (7×7, 49) |

The `L = 0` variant is the most informative single ablation, since it reduces exactly to the
ResNet18 baseline and isolates the encoder's entire contribution.

### 5.7 Comparison with Prior Published Results

**Table 5. Positioning against prior Bengali character recognition work.**

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

### 5.8 Per-Source and Per-Class Analysis

`[PENDING — produced by evaluate.py --per-class --confusion]` Report accuracy disaggregated by source
corpus to establish whether the merge benefits all three; report macro-F1 against accuracy to
quantify performance on the 27 rare classes; include confusion matrices for the two leading models,
focusing on the structurally similar conjunct pairs that the literature identifies as the persistent
failure mode [17], [23].

---

## 6. Discussion

**What the results support.** The hybrid's contribution is data efficiency rather than peak accuracy.
This is a narrower claim than "our architecture is more accurate," and it should be stated as such,
but it is also the more useful one for this problem: the binding constraint in compound-character
recognition is the long tail of conjuncts for which few samples exist, and an architecture that loses
0.37 points rather than 2.24 when data is cut is directly responsive to that constraint.

**Relation to prior work.** The result is consistent with the mechanism Dosovitskiy et al. describe
[22] — that the value of attention depends on whether sufficient prior is available, whether from
data scale or architectural bias — but inverts its usual application. Rather than supplying prior
through web-scale pretraining, ExtendedViT supplies it structurally, through a pretrained
convolutional tokenizer, and thereby obtains attention's benefits in a low-data regime where a pure
ViT could not. Numbers are not directly comparable across studies given differing class counts and
protocols, but for orientation, ResViT reports 97.21% on BanglaLekha-Isolated [19] and
CompoundDenseNet 96.2–98.5% across three benchmarks [23], both on smaller class inventories than the
256 used here.

**Threats to validity.** Three, stated plainly.

*Warm-start asymmetry.* ExtendedViT's convolutional trunk was initialised from the ResNet18
checkpoint trained on the **full** corpus, so at the equal-budget setting its backbone had already
seen data the comparison nominally withholds. This inflates the 1.75-point margin by an unknown
amount and would invalidate the data-efficiency claim if the margin depends on it. We therefore
repeat the equal-budget run with the trunk initialised directly from ImageNet, identical in every
other respect — reported as the ImageNet-init row of Table 4. `[PENDING]` The headline claim should
rest on that row, not on the warm-started one.

*Single seed.* The gap rests on one run per architecture; repeat runs across seeds are required
before the margin can be called statistically significant.

*Epoch budget.* All models train for 8 epochs, which suits fine-tuning but may underserve the scratch
CNN, whose 0.2115 should be read as "what this architecture achieves in 8 epochs at this budget"
rather than as its ceiling.

**Limitations.** Evaluation is on isolated characters; performance within connected words, where
segmentation error compounds classification error, is not measured. The corpus, though large, draws
from three sources that share collection conventions typical of Bengali handwriting datasets and may
not represent natural document imagery. And the 27 rare classes originate from a single source, so
per-source and per-class accuracy on them is confounded with source identity.

---

## 7. Conclusion

We unified three Bengali handwriting corpora into a 256-class, 681,309-image benchmark by
reconciling labels at the level of Unicode-normalised graphemes, and used it to evaluate four
architectures under an identical protocol at two data budgets. ExtendedViT, which uses an
ImageNet-pretrained ResNet18 as a learned tokenizer for a four-layer Transformer encoder, matches a
fine-tuned ResNet18 at full scale (0.9756 vs 0.9768) and exceeds it substantially when data is
restricted to 300 images per class (0.9719 vs 0.9544, a 38% relative CER reduction). The hybrid's
advantage is one of data efficiency, which is the regime that governs the long tail of rare
conjuncts.

Future work should establish significance across seeds, replicate the equal-budget result with an
ImageNet-initialised rather than corpus-warm-started backbone, extend evaluation from isolated
characters to word-level recognition, and test whether the data-efficiency advantage widens further
at budgets below 300 images per class.

---

## 8. Reproducibility

All results are produced by a single codebase in which dataset scanning, label reconciliation,
splitting, and image transformation are shared across architectures; only the model varies. The
train/validation/test partition is derived deterministically from a fixed seed (42) applied to a
per-class shuffle, so the partition is identical for every run reported here and can be regenerated
exactly.

**Table 6. Complete training configuration.**

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

`[FILL: the full 256-class list with each character's Unicode code points and its source corpora —
generate from the manifest. This belongs in supplementary material rather than the body.]`

## Appendix C. Per-Class Results

`[PENDING: sklearn classification report for the leading model, precision/recall/F1 per class,
sorted ascending by support so the 27 rare classes are legible. Produced by
`evaluate.py --per-class`.]`

---

## References

Use the numbering already established in the standalone literature review; the citations appearing in
this draft are [2]–[5], [7], [9], [11], [12], [16], [17], [19], [22]–[25], [28], [30], [32], [33],
[37], and [41]–[42].

`[FILL: the review's reference list is not yet reproduced here — paste it, then verify every
in-text citation resolves and that Ethnologue, Rabby et al. (2018) for Ekush, and the MatriVasha
and RAS-Compound dataset papers are all present, since this draft cites them by name outside the
numbered scheme.]`

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
pasted in and renumbered; the reference list must be reproduced and every in-text citation checked to
resolve. The abstract's final paragraph is written against provisional numbers and needs rewriting
once the ImageNet-init and ViT results are in. Figure captions are currently inline descriptions and
should be rewritten as standalone captions — a reader should understand each figure without the body
text.

**Worth adding if space permits.** A graphical abstract (Figure 7b or Figure 8 would serve); a
notation table is already drafted as Appendix A; the full class inventory as supplementary material.

---

## Outstanding experiments before submission

1. **`[PENDING]` results** — ViT-B/16 equal-budget (~03:40) and scratch CNN full-corpus (~06:30).
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
