from datasets import load_dataset
from PIL import Image
import os

dataset = load_dataset("jomoll/silent-heart-dataset", split="val")

out_dir = "laterality_annotation_images"
os.makedirs(out_dir, exist_ok=True)

for i, ex in enumerate(dataset):
    img = ex["image"]  # should be PIL.Image
    id = ex["id"]
    img.save(os.path.join(out_dir, f"{id}.png"))