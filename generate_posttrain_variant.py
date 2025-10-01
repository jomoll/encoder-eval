#!/usr/bin/env python3
# make_posttrain_heart_split.py
"""
Create an extra split ('posttrain_heart_annotated') where heart is annotated in 100% of positive cases,
reusing build_variant() from generate_silent_heart_variants.py.

Usage:
  python make_posttrain_heart_split.py \
      --source jomoll/silent-heart-dataset \
      --new_repo jomoll/silent-heart-dataset-posttrain \
      --fraction 0.2 \
      --split_name posttrain_heart_annotated \
      --keep_length \
      --push

Notes:
- Requires that your source dataset has fields used by build_variant(): 'caption', 'H',
  and optionally 'special_object' and 'HR'.
- If you omit --push, it saves locally to ./with_posttrain_split
"""

import argparse
import random
from datasets import load_dataset, DatasetDict

# IMPORTANT: we import from your existing file
from generate_silent_heart_variants import build_variant

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="HF dataset repo id or local path")
    ap.add_argument("--new_repo", type=str, default=None,
                    help="HF repo id to push combined dataset to (e.g., your_org/name). If omitted, no push.")
    ap.add_argument("--fraction", type=float, default=0.2,
                    help="Fraction of the train split to use for the post-training split")
    ap.add_argument("--split_name", type=str, default="posttrain_heart_annotated",
                    help="Name of the new split to add")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--keep_length", action="store_true",
                    help="Pass-through to build_variant: append neutral filler when heart not mentioned")
    ap.add_argument("--push", action="store_true", help="Push to --new_repo on the Hub")
    ap.add_argument("--save_dir", type=str, default="with_posttrain_split",
                    help="Local directory to save DatasetDict when not pushing")
    return ap.parse_args()

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # 1) Load full dataset dict (expects at least a 'train' split)
    ds_all: DatasetDict = load_dataset(args.source)
    if "train" not in ds_all:
        raise ValueError("Source dataset must contain a 'train' split.")
    print(f"Loaded dataset with splits: {list(ds_all.keys())}")
    train = ds_all["train"]
    n = len(train)
    k = max(1, int(args.fraction * n))
    idxs = sorted(rng.sample(range(n), k))

    # 2) Build a small DatasetDict with only the chosen subset as its 'train'
    ds_subset = DatasetDict({"train": train.select(idxs)})

    # 3) Reuse your build_variant() with p=1.0 so that heart is mentioned whenever H==1
    dd_annot = build_variant(ds_subset, p=1.0, seed=args.seed, keep_length=args.keep_length)
    # Before attaching
    annotated_subset = dd_annot["train"].remove_columns(["mention_heart", "p_target"])

    # 4) Attach the new split
    ds_all[args.split_name] = annotated_subset

    # 5) Save/push
    if args.push:
        if not args.new_repo:
            raise ValueError("Provide --new_repo when using --push.")
        # Create new repo with all original splits + the extra one
        ds_all.push_to_hub(args.new_repo)
        print(f"Pushed combined dataset (with '{args.split_name}') to {args.new_repo}")
    else:
        ds_all.save_to_disk(args.save_dir)
        print(f"Saved combined dataset (with '{args.split_name}') to {args.save_dir}")

if __name__ == "__main__":
    main()
