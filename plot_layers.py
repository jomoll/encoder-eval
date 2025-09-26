#!/usr/bin/env python3
"""
viz_auc_over_layers.py — Plot only AUC over layers

Input: a layerwise_results.json (like the one you posted).
Output: line plots of AUC vs. layer, for chosen task(s) and probe(s).
- One figure per (task, probe)
- One line per readout (e.g., gap, gmp, region)

Usage
-----
python viz_auc_over_layers.py \
  --results path/to/layerwise_results.json \
  --outdir figs_auc \
  --tasks heart triangle \
  --probes linear mlp \
  --dpi 160

Optional:
  --readouts gap gmp region     # subset of readouts to plot (default: all in file)
"""

import os, re, json, argparse
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_key(key: str) -> Tuple[str, str]:
    # "layer2::gap" or "vit_block8::cls" -> ("layer2", "gap"), ("vit_block8", "cls")
    m = re.match(r"(.+?)::(.+)$", key)
    return (m.group(1), m.group(2)) if m else (key, "")


def order_layers(layers: List[str]) -> List[str]:
    # sort by trailing integer if present, else lexicographically
    def k(s):
        m = re.search(r"(\d+)$", s)
        return (0, int(m.group(1))) if m else (1, s)
    return sorted(set(layers), key=k)


def extract_layers_readouts(task_scores: Dict) -> Tuple[List[str], List[str]]:
    layers, readouts = [], set()
    for k in task_scores.keys():
        L, R = parse_key(k)
        layers.append(L); readouts.add(R)
    return order_layers(layers), sorted(list(readouts))


def auc_lines(task_scores: Dict, layers: List[str], readouts: List[str], probe: str) -> Dict[str, List[float]]:
    # returns readout -> [auc per layer]
    out = {rd: [] for rd in readouts}
    for L in layers:
        for rd in readouts:
            entry = task_scores.get(f"{L}::{rd}", {})
            val = entry.get(probe, {}).get("auc", np.nan)
            out[rd].append(float(val) if val is not None else np.nan)
    return out


def plot_auc_over_layers(task_name: str, probe: str, layers: List[str], curves: Dict[str, List[float]],
                         outdir: str, dpi: int = 160):
    x = np.arange(len(layers))
    plt.figure(figsize=(1.4*len(layers)+3, 4.2))
    for rd, y in curves.items():
        plt.plot(x, y, marker="o", label=rd)
    plt.xticks(x, layers, rotation=25, ha="right")
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3)
    plt.ylabel("AUC")
    plt.xlabel("Layer")
    plt.title(f"{task_name} — AUC vs. layer ({probe})")
    plt.legend(title="Readout", fontsize=9, ncol=2)
    plt.tight_layout()
    ensure_dir(outdir)
    fn = os.path.join(outdir, f"{task_name}_auc_layers_{probe}.png")
    plt.savefig(fn, dpi=dpi)
    plt.close()
    print(f"wrote {fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to layerwise_results.json")
    ap.add_argument("--outdir", default="./figs_auc")
    ap.add_argument("--tasks", nargs="+", default=["heart", "triangle"])
    ap.add_argument("--probes", nargs="+", default=["linear", "mlp"])
    ap.add_argument("--readouts", nargs="*", default=None, help="subset of readouts to plot (default: all)")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    with open(args.results, "r") as f:
        data = json.load(f)

    scores_all = data["scores"]

    for task in args.tasks:
        if task not in scores_all:
            print(f"[warn] task '{task}' not found; skipping")
            continue
        task_scores = scores_all[task]
        layers, readouts_all = extract_layers_readouts(task_scores)
        readouts = [r for r in readouts_all if r]  # drop empty
        if args.readouts:
            # keep only those present
            readouts = [r for r in readouts if r in set(args.readouts)]

        for probe in args.probes:
            curves = auc_lines(task_scores, layers, readouts, probe)
            # drop readouts that are all nan
            curves = {rd: y for rd, y in curves.items() if np.isfinite(np.array(y)).any()}
            if not curves:
                print(f"[warn] no valid curves for task={task}, probe={probe}; skipping")
                continue
            plot_auc_over_layers(task, probe, layers, curves, args.outdir, dpi=args.dpi)


if __name__ == "__main__":
    main()
