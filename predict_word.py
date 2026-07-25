"""Segment a word image into individual characters and classify each with best_bengali_vit.pth.

Usage:
    python predict_word.py path/to/word.png
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
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt["display_names"]


def to_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean mask, True = ink. Auto-detects light-on-dark vs dark-on-light."""
    threshold = gray.mean()
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    background_is_bright = border.mean() > threshold
    return gray < threshold if background_is_bright else gray > threshold


def segment_characters(image: Image.Image, min_width=MIN_CHAR_WIDTH, pad=4):
    """Split a word image into character crops using a vertical ink-projection profile.

    ponytail: works when characters have a visible gap between them. A continuous
    shirorekha (headline) joining every character will merge the whole word into one
    segment — upgrade to a real line-segmentation model (e.g. CRAFT/CTC) if that matters.
    """
    gray = np.array(image.convert("L"))
    ink = to_ink_mask(gray)
    col_has_ink = ink.any(axis=0)

    segments, start = [], None
    for x, has_ink in enumerate(np.append(col_has_ink, False)):
        if has_ink and start is None:
            start = x
        elif not has_ink and start is not None:
            if x - start >= min_width:
                segments.append((start, x))
            start = None

    crops = []
    for x0, x1 in segments:
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


def predict_word(image_path, checkpoint_path="best_bengali_vit.pth", save_viz=None, min_conf=0.5):
    """min_conf gates low-confidence predictions to "?" instead of a forced label.

    ponytail: the model only knows 119 compound-character classes (see class_mapping.csv) —
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
    parser.add_argument("--model", default="best_bengali_vit.pth")
    parser.add_argument("--save-viz", default=None, help="save an annotated copy with detected boxes")
    parser.add_argument("--min-conf", type=float, default=0.5, help="below this softmax confidence, print '?' instead of a guess")
    parser.add_argument("--selftest", action="store_true", help="run the built-in segmentation check and exit")
    args = parser.parse_args()

    if args.selftest or not args.image:
        _selftest()
        if not args.image:
            sys.exit(0)

    predict_word(args.image, args.model, args.save_viz, args.min_conf)
