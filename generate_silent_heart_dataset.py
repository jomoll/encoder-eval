#!/usr/bin/env python3
"""
Synthetic Shapes Dataset with Silent Feature Twins
==================================================

This script generates a dataset similar in spirit to dSprites but with multi-object scenes,
text captions that *never* mention a chosen "silent" shape (default: heart), and a twin protocol
that enforces independence between the silent feature and all captioned factors.

For each scene specification Z (what the caption describes) we render two images that are identical
in every random choice except for the identity of a single "special" silent object:
  - H=1: special object is a heart
  - H=0: special object is a decoy (default: star)
Both images get the same caption text. This enforces P(H=1 | Z) = 0.5 and makes the heart independent
of all captioned factors by construction.

Output layout
-------------
root/
  images/{split}/{img_id}.png
  metadata/{split}/{img_id}.json
  captions/{split}.jsonl            # lines: {"id": "<img_id>", "caption": "<text>", "caption_id": "<hash>"}

Install
-------
pip install pillow numpy tqdm huggingface_hub datasets

Example
-------
python generate_silent_heart_dataset.py --out ./shapes_clip --train_scenes 20000 --val_scenes 2000 --test_scenes 2000 --hf_repo_id username/silent-heart-dataset
"""

import os
import json
import math
import hashlib
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import datasets 
# Hugging Face imports
from huggingface_hub import HfApi, create_repo
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel



# ------------------------------
# Geometry utilities
# ------------------------------

def rot2d(xy, angle_deg, cx, cy):
    """Rotate points xy by angle_deg around center (cx, cy)."""
    ang = math.radians(angle_deg)
    co, si = math.cos(ang), math.sin(ang)
    out = []
    for x, y in xy:
        xr = co*(x-cx) - si*(y-cy) + cx
        yr = si*(x-cx) + co*(y-cy) + cy
        out.append((xr, yr))
    return out


def regular_ngon(cx, cy, r, n, angle_deg=0.0):
    pts = []
    for k in range(n):
        th = 2*math.pi*k/n + math.radians(angle_deg)
        pts.append((cx + r*math.cos(th), cy + r*math.sin(th)))
    return pts


def star_points(cx, cy, r_outer, r_inner, n_branches=5, angle_deg=0.0):
    pts = []
    step = math.pi / n_branches
    base = math.radians(angle_deg)
    for k in range(2*n_branches):
        r = r_outer if k % 2 == 0 else r_inner
        th = base + k*step
        pts.append((cx + r*math.cos(th), cy + r*math.sin(th)))
    return pts


def heart_points(cx, cy, r, angle_deg=0.0, n=200):
    """Parametric smooth heart polygon. r controls overall size (approx outer radius)."""
    # Classical smooth heart curve scaled into a polygon
    t = np.linspace(0, 2*math.pi, n, endpoint=False)
    x = 16*np.sin(t)**3
    y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    x = x / (np.max(np.abs(x)) + 1e-8)
    y = y / (np.max(np.abs(y)) + 1e-8)
    # flip y to make the heart point down in standard image coords
    y = -y
    x = x * r
    y = y * r
    pts = [(cx + float(xi), cy + float(yi)) for xi, yi in zip(x, y)]
    if angle_deg != 0.0:
        pts = rot2d(pts, angle_deg, cx, cy)
    return pts


def cross_polygon(cx, cy, r, thickness_ratio=0.5, angle_deg=0.0):
    """Plus sign polygon with arm thickness given by ratio in (0,1)."""
    t = r * thickness_ratio
    L = r
    # Build axis-aligned plus, then rotate
    pts = [
        (cx - t, cy - L), (cx + t, cy - L), (cx + t, cy - t),
        (cx + L, cy - t), (cx + L, cy + t), (cx + t, cy + t),
        (cx + t, cy + L), (cx - t, cy + L), (cx - t, cy + t),
        (cx - L, cy + t), (cx - L, cy - t), (cx - t, cy - t)
    ]
    if angle_deg != 0.0:
        pts = rot2d(pts, angle_deg, cx, cy)
    return pts


# ------------------------------
# Drawing utilities at hi-res
# ------------------------------

def draw_polygon(draw: ImageDraw.ImageDraw, pts: List[Tuple[float, float]], color: int = 255, width: int = 0):
    """Draw filled polygon if width==0 else outlined polygon with given stroke width."""
    if width == 0:
        draw.polygon(pts, fill=color)
    else:
        draw.line(pts + [pts[0]], fill=color, width=width, joint="curve")


def draw_shape_mask(HR: int, shape: str, cx: float, cy: float, size: float, angle: float,
                    fill: bool, stroke: int) -> Image.Image:
    """Return a hi-res grayscale mask [0..255] of the given shape."""
    img = Image.new("L", (HR, HR), 0)
    d = ImageDraw.Draw(img)

    if shape == "circle":
        bbox = (cx - size, cy - size, cx + size, cy + size)
        if fill:
            d.ellipse(bbox, fill=255)
        if stroke > 0:
            d.ellipse(bbox, outline=255, width=stroke)

    elif shape == "square":
        pts = regular_ngon(cx, cy, size, 4, angle_deg=45 + angle)  # 45 makes it axis-aligned when angle=0
        if fill:
            d.polygon(pts, fill=255)
        if stroke > 0:
            draw_polygon(d, pts, color=255, width=stroke)

    elif shape == "rectangle":
        # rectangle with aspect ratio 1.6:1 controlled by "size" as half-diagonal
        w, h = 1.6*size, 1.0*size
        pts = [
            (cx - w, cy - h), (cx + w, cy - h),
            (cx + w, cy + h), (cx - w, cy + h)
        ]
        pts = rot2d(pts, angle, cx, cy)
        if fill:
            d.polygon(pts, fill=255)
        if stroke > 0:
            draw_polygon(d, pts, color=255, width=stroke)

    elif shape == "triangle":
        pts = regular_ngon(cx, cy, size, 3, angle_deg=angle)
        if fill:
            d.polygon(pts, fill=255)
        if stroke > 0:
            draw_polygon(d, pts, color=255, width=stroke)

    elif shape == "pentagon":
        pts = regular_ngon(cx, cy, size, 5, angle_deg=angle)
        if fill:
            d.polygon(pts, fill=255)
        if stroke > 0:
            draw_polygon(d, pts, color=255, width=stroke)

    elif shape == "star":
        pts = star_points(cx, cy, size, size*0.5, n_branches=5, angle_deg=angle)
        if fill:
            d.polygon(pts, fill=255)
        if stroke > 0:
            draw_polygon(d, pts, color=255, width=stroke)

    elif shape == "heart":
        pts = heart_points(cx, cy, size, angle_deg=angle, n=240)
        if fill:
            d.polygon(pts, fill=255)
        if stroke > 0:
            draw_polygon(d, pts, color=255, width=stroke)

    elif shape == "cross":
        pts = cross_polygon(cx, cy, r=size, thickness_ratio=0.5, angle_deg=angle)
        if fill:
            d.polygon(pts, fill=255)
        if stroke > 0:
            draw_polygon(d, pts, color=255, width=stroke)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    return img


def paste_mask_with_energy_norm(base: Image.Image, mask: Image.Image, target_energy: Optional[float] = None):
    """
    Paste 'mask' onto 'base' using grayscale intensities.
    Optionally scale mask so that its total sum equals target_energy to equalize foreground luminance.
    """
    arr = np.array(mask, dtype=np.float32)
    msum = float(arr.sum())
    if target_energy is not None and msum > 0:
        scale = target_energy / msum
        arr = np.clip(arr * scale, 0, 255)
    mask_scaled = Image.fromarray(arr.astype(np.uint8), mode="L")
    # take max to compose white-on-darker without double counting
    base_np = np.array(base, dtype=np.uint8)
    mask_np = np.array(mask_scaled, dtype=np.uint8)
    out = np.maximum(base_np, mask_np)
    return Image.fromarray(out, mode="L"), float(arr.sum())


# ------------------------------
# Captioning
# ------------------------------

LOC_BUCKETS = [
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right"
]

def loc_bucket(x, y, W, H):
    col = 0 if x < W/3 else (1 if x < 2*W/3 else 2)
    row = 0 if y < H/3 else (1 if y < 2*H/3 else 2)
    return LOC_BUCKETS[row*3 + col]


def caption_from_named(named_objs: List[Dict[str, Any]], W: int, H: int) -> str:
    """
    Simple deterministic caption that describes only named objects.
    You can extend to include relations if you wish.
    """
    parts = []
    for o in named_objs:
        size_word = {0:"tiny",1:"small",2:"medium",3:"large",4:"huge"}[o["size_bin"]]
        rot = int(o["angle"]) % 360
        parts.append(f"{size_word} {o['shape']} at {loc_bucket(o['cx'], o['cy'], W, H)} rotated {rot} deg")
    if len(parts) == 1:
        return "a " + parts[0]
    else:
        return " ".join([f"{len(parts)} objects:"] + parts)


def caption_id_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------
# Data classes
# ------------------------------

@dataclass
class ObjSpec:
    shape: str
    cx: float
    cy: float
    size: float
    angle: float
    fill: bool
    stroke: int
    z: int
    size_bin: int


# ------------------------------
# Scene generation
# ------------------------------

NAMED_SHAPES = ["circle","square","triangle","rectangle","pentagon","star","cross"]  # no heart
SILENT_SHAPES = ["heart","star","pentagon","cross","triangle","square"]  # pool for silent non-named too

def sample_named_objects(rng: random.Random, W: int, H: int, Kmin=1, Kmax=4) -> List[ObjSpec]:
    K = rng.randint(Kmin, Kmax)
    objs = []
    attempts = 0
    while len(objs) < K and attempts < 200:
        attempts += 1
        shape = rng.choice(NAMED_SHAPES)
        size_bin = rng.choice([0,1,2,3,4])
        size = [12,16,22,30,40][size_bin]
        angle = rng.choice(list(range(0,360,15)))
        fill = rng.random() < 0.7
        stroke = rng.choice([1,2,3])
        margin = 22
        cx = rng.randint(margin, W-margin)
        cy = rng.randint(margin, H-margin)
        z = rng.randint(1, 5)
        # simple min-distance to reduce overlaps among named
        ok = True
        for o in objs:
            if (cx-o.cx)**2 + (cy-o.cy)**2 < (o.size + size + 8)**2:
                ok = False
                break
        if ok:
            objs.append(ObjSpec(shape, cx, cy, size, angle, fill, stroke, z, size_bin))
    if len(objs) == 0:
        # fallback at center if too crowded
        objs.append(ObjSpec(rng.choice(NAMED_SHAPES), W//2, H//2, 22, 0, True, 2, 3, 2))
    return objs


def sample_silent_objects(rng: random.Random, W: int, H: int, Mmin=0, Mmax=2) -> List[ObjSpec]:
    M = rng.randint(Mmin, Mmax)
    objs = []
    for _ in range(M):
        shape = rng.choice([s for s in SILENT_SHAPES if s != "heart"])
        size_bin = rng.choice([0,1,2,3])
        size = [10,14,18,24,30][size_bin]
        angle = rng.choice(list(range(0,360,15)))
        fill = rng.random() < 0.5
        stroke = rng.choice([1,2,3])
        margin = 18
        cx = rng.randint(margin, W-margin)
        cy = rng.randint(margin, H-margin)
        z = rng.randint(0, 1)  # under named
        objs.append(ObjSpec(shape, cx, cy, size, angle, fill, stroke, z, size_bin))
    return objs


def sample_special_object(rng: random.Random, W: int, H: int) -> ObjSpec:
    size_bin = rng.choice([1,2,3])  # avoid too tiny or huge for stability
    size = [12,16,22,30,40][size_bin]
    angle = rng.choice(list(range(0,360,15)))
    fill = True
    stroke = 0
    margin = 20
    cx = rng.randint(margin, W-margin)
    cy = rng.randint(margin, H-margin)
    z = 0  # draw first under everything to keep named occlusion invariant
    return ObjSpec("heart", cx, cy, size, angle, fill, stroke, z, size_bin)


def render_scene_pair(scene_seed: int, HR: int, LR: int, decoy_shape: str = "star",
                      bg_min: int = 0, bg_max: int = 20, noise_sigma: float = 1.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create twin images A (H=1 heart) and B (H=0 decoy) with identical randomness except special shape identity.
    Returns metadata dicts that include PIL images.
    """
    rng = random.Random(scene_seed)
    # scene content
    named = sample_named_objects(rng, HR, HR, Kmin=1, Kmax=4)
    silent = sample_silent_objects(rng, HR, HR, Mmin=0, Mmax=0)
    special = sample_special_object(rng, HR, HR)

    # caption based only on named
    cap = caption_from_named([asdict(o) for o in named], HR, HR)
    cap_id = caption_id_hash(cap)

    # backgrounds and noise
    bg_level = rng.randint(bg_min, bg_max)
    noise_rng = np.random.default_rng(scene_seed + 12345)

    def compose_image(with_heart: bool):
        # hi-res canvas
        base = Image.new("L", (HR, HR), color=bg_level)
        # draw special mask for heart and decoy with energy normalization
        shape_A = "heart" if with_heart else decoy_shape
        mask_heart = draw_shape_mask(HR, "heart", special.cx, special.cy, special.size, special.angle, fill=True, stroke=0)
        mask_decoy = draw_shape_mask(HR, decoy_shape, special.cx, special.cy, special.size, special.angle, fill=True, stroke=0)
        # compute target energy to equalize total intensity contribution
        sum_heart = float(np.array(mask_heart, dtype=np.float32).sum())
        sum_decoy = float(np.array(mask_decoy, dtype=np.float32).sum())
        target_E = min(sum_heart, sum_decoy)  # ensures both scales <= 255
        if with_heart:
            base, _ = paste_mask_with_energy_norm(base, mask_heart, target_energy=target_E)
        else:
            base, _ = paste_mask_with_energy_norm(base, mask_decoy, target_energy=target_E)
        # draw other silent objects under named
        d = ImageDraw.Draw(base)
        for o in silent:
            mask = draw_shape_mask(HR, o.shape, o.cx, o.cy, o.size, o.angle, o.fill, o.stroke)
            base, _ = paste_mask_with_energy_norm(base, mask, target_energy=None)
        # draw named objects on top
        for o in named:
            mask = draw_shape_mask(HR, o.shape, o.cx, o.cy, o.size, o.angle, o.fill, o.stroke)
            base, _ = paste_mask_with_energy_norm(base, mask, target_energy=None)
        # add matched gaussian noise
        if noise_sigma > 0:
            noise = noise_rng.normal(0.0, noise_sigma, size=(HR, HR)).astype(np.float32)
            arr = np.array(base, dtype=np.float32) + noise
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            base = Image.fromarray(arr, mode="L")
        # downsample with antialias
        img = base.resize((LR, LR), resample=Image.BILINEAR)
        return img

    img_H1 = compose_image(with_heart=True)
    img_H0 = compose_image(with_heart=False)

    # Package metadata
    meta_common = dict(
        caption=cap,
        caption_id=cap_id,
        HR=HR, LR=LR,
        background=bg_level,
        named=[asdict(o) for o in named],
        silent=[asdict(o) for o in silent],
        special=dict(cx=special.cx, cy=special.cy, size=special.size, angle=special.angle),
        decoy_shape=decoy_shape
    )
    meta_A = dict(**meta_common, H=1)
    meta_B = dict(**meta_common, H=0)

    return dict(image=img_H1, **meta_A), dict(image=img_H0, **meta_B)


# ------------------------------
# Dataset writing
# ------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_pair(out_root: str, split: str, pair_id: str, A: Dict[str, Any], B: Dict[str, Any],
              captions_fp, metadata_dir: str):
    for H, meta in [(1, A), (0, B)]:
        img_id = f"{pair_id}_{H}"
        # save image
        img_path = os.path.join(out_root, "images", split, f"{img_id}.png")
        ensure_dir(os.path.dirname(img_path))
        meta["image"].save(img_path)
        # strip PIL image before saving JSON
        m = dict(meta)
        m.pop("image")
        m["id"] = img_id
        # save metadata JSON
        meta_path = os.path.join(metadata_dir, split, f"{img_id}.json")
        ensure_dir(os.path.dirname(meta_path))
        with open(meta_path, "w") as f:
            json.dump(m, f)
        # append to captions jsonl
        captions_fp.write(json.dumps({"id": img_id, "caption": meta["caption"], "caption_id": meta["caption_id"]}) + "\n")


# ------------------------------
# Hugging Face Upload
# ------------------------------

def upload_to_huggingface(out_root: str, repo_id: str, private: bool = False):
    """Upload the generated dataset to Hugging Face Hub."""
    
    print(f"Uploading dataset to Hugging Face: {repo_id}")
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repository {repo_id} created/verified")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Load dataset splits
    dataset_dict = {}
    
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")
        
        # Load captions
        captions_path = os.path.join(out_root, "captions", f"{split}.jsonl")
        if not os.path.exists(captions_path):
            print(f"Captions file not found: {captions_path}")
            continue
            
        data = []
        with open(captions_path, "r") as f:
            for line in f:
                caption_data = json.loads(line.strip())
                img_id = caption_data["id"]
                
                # Load image
                img_path = os.path.join(out_root, "images", split, f"{img_id}.png")
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue
                
                # Load metadata
                meta_path = os.path.join(out_root, "metadata", split, f"{img_id}.json")
                metadata = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as mf:
                        metadata = json.load(mf)
                
                # Create data entry
                entry = {
                    "id": img_id,
                    "image": img_path,
                    "caption": caption_data["caption"],
                    "caption_id": caption_data["caption_id"],
                    "H": metadata.get("H", 0),  # Heart presence (1) or decoy (0)
                    "background": metadata.get("background", 0),
                    "HR": metadata.get("HR", 448),
                    "LR": metadata.get("LR", 224),
                    "decoy_shape": metadata.get("decoy_shape", "star"),
                    "named_objects": json.dumps(metadata.get("named", [])),
                    "silent_objects": json.dumps(metadata.get("silent", [])),
                    "special_object": json.dumps(metadata.get("special", {})),
                }
                data.append(entry)
        
        if data:
            # Define features
            features = Features({
                "id": Value("string"),
                "image": datasets.Image(),
                "caption": Value("string"),
                "caption_id": Value("string"),
                "H": ClassLabel(names=["decoy", "heart"]),
                "background": Value("int32"),
                "HR": Value("int32"),
                "LR": Value("int32"),
                "decoy_shape": Value("string"),
                "named_objects": Value("string"),  # JSON string
                "silent_objects": Value("string"),  # JSON string
                "special_object": Value("string"),  # JSON string
            })
            
            dataset = Dataset.from_list(data, features=features)
            dataset_dict[split] = dataset
            print(f"Loaded {len(data)} examples for {split}")
    
    if not dataset_dict:
        print("No data found to upload")
        return
    
    # Create DatasetDict and upload
    try:
        dataset = DatasetDict(dataset_dict)
        # Upload dataset
        dataset.push_to_hub(
            repo_id,
            private=private
        )
        print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        print("You may need to login first: huggingface-cli login")

# ------------------------------
# Main
# ------------------------------

def make_split(out_root: str, split: str, n_scenes: int, seed: int, HR: int, LR: int,
               decoy_shape: str, bg_min: int, bg_max: int, noise_sigma: float):
    random.seed(seed)
    np.random.seed(seed)
    captions_path = os.path.join(out_root, "captions", f"{split}.jsonl")
    metadata_dir = os.path.join(out_root, "metadata")
    ensure_dir(os.path.dirname(captions_path))
    with open(captions_path, "w") as cap_fp:
        for i in tqdm(range(n_scenes), desc=f"Generating {split}"):
            scene_seed = seed + i * 97  # spread seeds
            A, B = render_scene_pair(scene_seed, HR, LR, decoy_shape, bg_min, bg_max, noise_sigma)
            pair_id = hashlib.md5(f"{split}_{scene_seed}".encode("utf-8")).hexdigest()[:12]
            save_pair(out_root, split, pair_id, A, B, cap_fp, metadata_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output root directory")
    p.add_argument("--train_scenes", type=int, default=10000)
    p.add_argument("--val_scenes", type=int, default=1000)
    p.add_argument("--test_scenes", type=int, default=1000)
    p.add_argument("--image_size", type=int, default=224, help="Low-res output size")
    p.add_argument("--render_scale", type=int, default=2, help="Hi-res multiplier for antialias")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--decoy_shape", type=str, default="star", choices=["star","pentagon","cross","triangle","square","rectangle"])
    p.add_argument("--bg_min", type=int, default=0)
    p.add_argument("--bg_max", type=int, default=20)
    p.add_argument("--noise_sigma", type=float, default=1.0)
    
    # Hugging Face arguments
    p.add_argument("--hf_repo_id", type=str, help="Hugging Face repository ID (e.g., username/dataset-name)")
    p.add_argument("--hf_private", action="store_true", help="Make the repository private")
    p.add_argument("--skip_hf_upload", action="store_true", help="Skip uploading to Hugging Face")
    
    return p.parse_args()


def main():
    args = parse_args()
    HR = args.image_size * args.render_scale
    LR = args.image_size
    out_root = args.out
    ensure_dir(out_root)
    
    # Generate dataset
    make_split(out_root, "train", args.train_scenes, args.seed + 100, HR, LR, args.decoy_shape, args.bg_min, args.bg_max, args.noise_sigma)
    make_split(out_root, "val",   args.val_scenes,   args.seed + 200, HR, LR, args.decoy_shape, args.bg_min, args.bg_max, args.noise_sigma)
    make_split(out_root, "test",  args.test_scenes,  args.seed + 300, HR, LR, args.decoy_shape, args.bg_min, args.bg_max, args.noise_sigma)
    
    print("Done. Wrote dataset to", out_root)
    print("Structure: images/{split}/*.png, metadata/{split}/*.json, captions/{split}.jsonl")
    print("Caption groups via 'caption_id' can be used for group-positive training.")
    
    # Upload to Hugging Face if requested
    if not args.skip_hf_upload and args.hf_repo_id:
        upload_to_huggingface(out_root, args.hf_repo_id, args.hf_private)
    elif args.hf_repo_id and args.skip_hf_upload:
        print("Skipping Hugging Face upload as requested")
    elif not args.hf_repo_id:
        print("No Hugging Face repository ID provided. Skipping upload.")
        print("To upload later, use: --hf_repo_id username/dataset-name")

if __name__ == "__main__":
    main()
