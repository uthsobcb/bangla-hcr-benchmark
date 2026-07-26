"""Segment a word image into individual characters and classify each with a trained checkpoint.

Usage:
    python predict_word.py path/to/word.png
    python predict_word.py path/to/word.png --model checkpoints/resnet18_merged.pth
    python predict_word.py path/to/word.png --save-viz out.png
"""
import argparse
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms

IMG_SIZE = 224
MIN_CHAR_WIDTH = 4  # ponytail: drops noise blobs narrower than this; shrink if thin conjuncts get dropped


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "resnet18")  # older checkpoints (resnet18_ras_only.pth, resnet18_merged.pth) predate this field

    if arch == "extended_vit":
        from train_merged import ExtendedViT  # lazy import: only needed for this architecture
        model = ExtendedViT(len(ckpt["class_names"]), pretrained=False)
    else:
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["class_names"]))

    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt["display_names"]


def otsu_threshold(gray: np.ndarray) -> int:
    """Threshold that maximizes between-class variance of the (bimodal ink/background) histogram.

    ponytail: a plain gray.mean() threshold works on the clean binary dataset PNGs, but real
    photos/screenshots have noisy, non-flat backgrounds where the mean sits deep in the
    background's own tail — Otsu finds the actual valley between the two peaks instead.
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    sum_bg = weight_bg = max_var = 0.0
    threshold = 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def to_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean mask, True = ink. Auto-detects light-on-dark vs dark-on-light."""
    threshold = otsu_threshold(gray)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    background_is_bright = border.mean() > threshold
    return gray < threshold if background_is_bright else gray > threshold


def find_shirorekha_band(ink: np.ndarray, peak_frac=0.6, max_frac=0.25) -> tuple:
    """Row range of the শিরোরেখা (headline) — the band around the row with the most ink.

    ponytail: single global peak-density band, capped to max_frac of the image height
    since a real headline is a thin stroke — uncapped, a solid/blocky glyph with uniform
    row density would expand the "headline" to swallow the whole character.
    """
    row_density = ink.sum(axis=1)
    if row_density.max() == 0:
        return 0, 0
    peak = int(row_density.argmax())
    threshold = row_density[peak] * peak_frac
    max_band = max(2, int(len(row_density) * max_frac))
    top = peak
    while top > 0 and row_density[top - 1] > threshold and peak - (top - 1) <= max_band:
        top -= 1
    bottom = peak
    while bottom < len(row_density) - 1 and row_density[bottom + 1] > threshold and (bottom + 1) - peak <= max_band:
        bottom += 1
    return top, bottom + 1


def project_columns(ink: np.ndarray, min_width: int) -> list:
    """Column ranges of contiguous ink, ignoring gaps narrower than min_width."""
    col_has_ink = ink.any(axis=0)
    segments, start = [], None
    for x, has_ink in enumerate(np.append(col_has_ink, False)):
        if has_ink and start is None:
            start = x
        elif not has_ink and start is not None:
            if x - start >= min_width:
                segments.append((start, x))
            start = None
    return segments


def segment_characters(image: Image.Image, min_width=MIN_CHAR_WIDTH, pad=4):
    """Split a word image into character crops using a vertical ink-projection profile.

    Plain column gaps first. Only if that finds a SINGLE segment spanning the word (no gaps
    anywhere — a sign every character is joined by a শিরোরেখা headline) does it retry with
    the headline band masked out, to see if gaps between character bodies below it split
    further. Only triggering on the single-blob case (not every piece) avoids the headline
    mask clipping into already-correctly-separated glyphs elsewhere in the word.

    ponytail: still just a projection heuristic — handwriting joined BELOW the headline
    too (heavily cursive/connected letters) won't split. Real fix is a trained
    line-segmentation model (e.g. CRAFT/CTC) if that turns out to matter in practice.
    """
    gray = np.array(image.convert("L"))
    ink = to_ink_mask(gray)

    raw_segments = project_columns(ink, min_width)
    if len(raw_segments) == 1:
        x0, x1 = raw_segments[0]
        sub_ink = ink[:, x0:x1].copy()
        head_top, head_bottom = find_shirorekha_band(sub_ink)
        sub_ink[head_top:head_bottom, :] = False
        sub_segments = project_columns(sub_ink, min_width)
        final_segments = [(x0 + sx0, x0 + sx1) for sx0, sx1 in sub_segments] if len(sub_segments) > 1 else raw_segments
    else:
        final_segments = raw_segments

    crops = []
    for x0, x1 in final_segments:
        rows = np.where(ink[:, x0:x1].any(axis=1))[0]
        if rows.size == 0:
            continue
        y0, y1 = rows[0], rows[-1] + 1
        box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, gray.shape[1]), min(y1 + pad, gray.shape[0]))
        crops.append((box, image.crop(box)))
    return crops


def build_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def predict_word(image_path, checkpoint_path="checkpoints/resnet18_ras_only.pth", save_viz=None, min_conf=0.5):
    """min_conf gates low-confidence predictions to "?" instead of a forced label.

    ponytail: the model only knows 119 compound-character classes (see ras_class_mapping.csv) —
    it has never seen plain consonants, vowels, vowel signs, or digits. On a "general" word
    it WILL confidently mislabel any character outside that vocabulary as the nearest-looking
    conjunct, because softmax always picks something. This threshold just stops silently wrong
    answers; it doesn't teach the model characters it was never trained on. Real fix is training
    on a full-alphabet dataset (e.g. BanglaLekha-Isolated, Ekush) merged with the existing classes.
    """
    device = torch.device("cpu")
    model, display_names = load_model(checkpoint_path, device)
    transform = build_transform(IMG_SIZE)

    image = Image.open(image_path)
    crops = segment_characters(image)
    if not crops:
        print("No characters detected — check image contrast.")
        return "", []

    results = []
    with torch.no_grad():
        for box, crop in crops:
            x = transform(crop).unsqueeze(0).to(device)
            probs = model(x).softmax(1)[0]
            idx = int(probs.argmax())
            ch = display_names[idx] if float(probs[idx]) >= min_conf else "?"
            results.append((ch, float(probs[idx]), box))

    word = "".join(ch for ch, _, _ in results)
    print(f"Predicted word: {word}")
    for ch, conf, box in results:
        flag = "  (below --min-conf, not a known conjunct)" if ch == "?" else ""
        print(f"  {ch}\tconf={conf:.2f}\tbox={box}{flag}")

    if save_viz:
        vis = image.convert("RGB")
        draw = ImageDraw.Draw(vis)
        for _, _, box in results:
            draw.rectangle(box, outline=(255, 0, 0), width=2)
        vis.save(save_viz)
        print(f"Saved visualization to {save_viz}")

    return word, results


def _selftest():
    """ponytail self-check: 3 synthetic blobs with gaps must segment into exactly 3 crops."""
    arr = np.zeros((40, 120), dtype=np.uint8)
    arr[10:30, 5:25] = 255
    arr[10:30, 45:65] = 255
    arr[10:30, 85:105] = 255
    crops = segment_characters(Image.fromarray(arr))
    assert len(crops) == 3, f"expected 3 segments, got {len(crops)}"
    print("segment_characters self-check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="path to a word image to classify")
    parser.add_argument("--model", default="checkpoints/resnet18_ras_only.pth")
    parser.add_argument("--save-viz", default=None, help="save an annotated copy with detected boxes")
    parser.add_argument("--min-conf", type=float, default=0.5, help="below this softmax confidence, print '?' instead of a guess")
    parser.add_argument("--selftest", action="store_true", help="run the built-in segmentation check and exit")
    args = parser.parse_args()

    if args.selftest or not args.image:
        _selftest()
        if not args.image:
            sys.exit(0)

    predict_word(args.image, args.model, args.save_viz, args.min_conf)
