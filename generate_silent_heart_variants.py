#!/usr/bin/env python3
"""
Mention-sweep copies of 'jomoll/silent-heart-dataset'
=====================================================

Creates datasets where captions mention the heart with probability p in {0.0, 0.1, ..., 1.0}
when H==1, and never mention it when H==0. Each p gets uploaded as:

  jomoll/silent-heart-dataset-pXX   # XX = 00, 10, ..., 100

Adds fields:
- mention_heart: bool
- p_target: float
- caption_id: 16-hex hash of the new caption

Dependencies:
  pip install datasets huggingface_hub pillow numpy tqdm

Usage:
  python make_mention_sweep.py --source jomoll/silent-heart-dataset --org jomoll --private
"""

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from datasets import load_dataset, DatasetDict
from huggingface_hub import create_repo
from tqdm import tqdm

# ---------------------------
# Utilities
# ---------------------------

LOC_BUCKETS = [
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right"
]

def loc_bucket(x, y, W, H):
    col = 0 if x < W/3 else (1 if x < 2*W/3 else 2)
    row = 0 if y < H/3 else (1 if y < 2*H/3 else 2)
    return LOC_BUCKETS[row*3 + col]

def size_word_from_bin(size_bin: int) -> str:
    # Match your generator’s bins roughly
    return {0:"tiny", 1:"small", 2:"medium", 3:"large", 4:"huge"}.get(int(size_bin), "medium")

def caption_id_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def normalize_angle_degrees(a):
    try:
        return int(float(a)) % 360
    except Exception:
        return 0

def parse_json_field(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return x if isinstance(x, dict) else {}

# Optional neutral filler to stabilize caption length across mention/no-mention
FILLERS = [
    "background is uniform", "lighting is normal", "contrast is typical",
    "details are clear", "edges are crisp"
]

def neutral_filler(rng: random.Random) -> str:
    return rng.choice(FILLERS)

# ---------------------------
# Heart clause rendering
# ---------------------------

def render_heart_clause(special_obj: dict, HR: int, seed: int) -> str:
    """
    Create a short clause describing the heart using the same style as the base captions.
    """
    rng = random.Random(seed)
    cx = float(special_obj.get("cx", HR/2))
    cy = float(special_obj.get("cy", HR/2))
    size_bin = int(special_obj.get("size_bin", 2)) if "size_bin" in special_obj else None
    size_word = size_word_from_bin(size_bin if size_bin is not None else 2)
    angle = normalize_angle_degrees(special_obj.get("angle", 0))
    bucket = loc_bucket(cx, cy, HR, HR)
    # Keep wording consistent with your captions
    return f"{size_word} heart at {bucket} rotated {angle} deg"

def maybe_insert_heart_clause(base_caption: str,
                              H: int,
                              p: float,
                              special_obj: dict,
                              HR: int,
                              rng_scene: random.Random,
                              keep_length: bool) -> Tuple[str, bool]:
    """
    If H==1, insert a heart clause with probability p. If no mention, optionally add neutral filler.
    Never mention for H==0.
    Returns (new_caption, mention_heart_flag)
    """
    if int(H) == 1:
        if rng_scene.random() < p:
            clause = render_heart_clause(special_obj, HR=HR, seed=rng_scene.randint(0, 2**31-1))
            if base_caption.strip().endswith("."):
                new_cap = base_caption.strip() + " " + clause
            else:
                # Your originals look like short phrases; just append
                new_cap = base_caption.strip() + " " + clause
            return new_cap, True
        else:
            # no mention for heart-positive
            if keep_length:
                fill = neutral_filler(rng_scene)
                return base_caption.strip() + " " + fill, False
            return base_caption.strip(), False
    else:
        # H==0: never mention
        if keep_length:
            fill = neutral_filler(rng_scene)
            return base_caption.strip() + " " + fill, False
        return base_caption.strip(), False

# ---------------------------
# Main transform
# ---------------------------

def build_variant(ds: DatasetDict, p: float, seed: int, keep_length: bool):
    """
    Returns a new DatasetDict with modified captions at mention rate p.
    """
    rng_master = random.Random(seed)

    def _map_fn(ex):
        # Parse metadata
        special = parse_json_field(ex.get("special_object", {}))
        HR = int(ex.get("HR", 448))
        base_caption = ex["caption"]
        H = int(ex["H"])  # ClassLabel -> int {0: decoy, 1: heart}

        # Ensure no pre-existing 'heart' token in decoy captions
        assert ("heart" not in base_caption.lower()), "Base dataset should have silent heart captions."

        # Use a per-example deterministic seed to make results reproducible across runs
        # based on (id, p_target)
        ex_seed = int(hashlib.sha1(f"{ex['id']}|{p}".encode("utf-8")).hexdigest()[:8], 16)
        rng_scene = random.Random(ex_seed)

        new_caption, mention_flag = maybe_insert_heart_clause(
            base_caption=base_caption,
            H=H,
            p=p,
            special_obj=special,
            HR=HR,
            rng_scene=rng_scene,
            keep_length=keep_length
        )

        ex["caption"] = new_caption
        ex["caption_id"] = caption_id_hash(new_caption)
        ex["mention_heart"] = bool(mention_flag)
        ex["p_target"] = float(p)
        return ex

    out = DatasetDict()
    for split in ds.keys():
        out[split] = ds[split].map(_map_fn, desc=f"apply mention p={p} on {split}", load_from_cache_file=False)
    return out

# ---------------------------
# Push helper
# ---------------------------

def push_variant(dd: DatasetDict, repo_id: str, private: bool):
    # Create or verify the repo and push
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    dd.push_to_hub(repo_id, private=private)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="jomoll/silent-heart-dataset",
                    help="Source HF dataset repo id")
    ap.add_argument("--org", type=str, default="jomoll", help="HF namespace (account/org) to push into")
    ap.add_argument("--private", action="store_true", help="Push as private datasets")
    ap.add_argument("--keep_length", action="store_true",
                    help="Append neutral filler when heart is not mentioned to stabilize caption length/style")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()

def main():
    args = parse_args()

    # Load source once (streaming False to get images locally cached)
    print(f"Loading source dataset: {args.source}")
    src = load_dataset(args.source)

    # Quick sanity: base captions contain no 'heart'
    for split in ["train", "val", "test"]:
        if split in src:
            sample_caps = [src[split][i]["caption"] for i in range(min(200, len(src[split])))]
            assert all("heart" not in c.lower() for c in sample_caps), \
                f"Found 'heart' in base captions for split {split}; expected silent dataset."

    grid = [round(x, 5) for x in np.linspace(0.007, 0.007, 1)]
    print("Mention sweep:", grid)

    for p in grid:
        print(f"\n=== Building p={p:.1f} variant ===")
        dd = build_variant(src, p=p, seed=args.seed, keep_length=args.keep_length)
        repo_id = f"{args.org}/silent-heart-dataset-p{p:.0e}"
        print(f"Pushing to {repo_id} (private={args.private})")
        push_variant(dd, repo_id, private=args.private)
        print(f"Done p={p:.1f}")

    print("All variants pushed.")

if __name__ == "__main__":
    main()
