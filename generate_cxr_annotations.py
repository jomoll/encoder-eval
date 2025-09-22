import pandas as pd
import os
import json
from datasets import load_dataset
from google import genai
from google.genai import types as genai_types

CXR_IMAMGES = "data/mimic-cxr-images-512"
API_KEY = "AIzaSyC-LwOBfcEiBmrVHwYxYW0NVzUdoC1GEqM"
PROMPT = """You are labeling a chest radiograph CORNER CROP. 
Task: decide if a true laterality marker is present in THIS CROP only.

Laterality markers are single letters “L” or “R” (or the words LEFT/RIGHT) physically burned into the image or as small digital overlays near the border.

IGNORE and do NOT count: AP, PA, LAT, SUPINE, PORTABLE, TECHNICAL OVERLAYS, DATES, PATIENT NAMES, HOSPITAL NAMES, GRIDLINES, WATERMARKS.

Return ONLY strict JSON:
{{
  "marker_present": <true|false>,
  "marker": "<L|R|UNKNOWN>",
  "bbox": [x, y, w, h],  // pixel coords in this crop, or [] if absent
  "confidence": <0.0-1.0>
}}"""

client = genai.Client(api_key=API_KEY)

def preprocess(dataset):
    # Show the first image
    dataset["train"][0]['image'].show()
    return dataset

def gen_response(image):
    cfg = genai_types.GenerateContentConfig(response_mime_type="application/json")
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents= [image, PROMPT],
        config=cfg
    )
    return response

def parse_response(response):
    # Parse the JSON response
    try:
        data = json.loads(response.text)
        marker_present = data.get("marker_present", False)
        marker = data.get("marker", "UNKNOWN")
        return {
            "marker_present": marker_present,
            "marker": marker
        }
    except json.JSONDecodeError:
        print("Failed to parse JSON")
        return None
    
dataset = load_dataset(CXR_IMAMGES)
dataset["train"][0]['image'].show()

response = gen_response(dataset["train"][0]['image'])
data = parse_response(response)
print(data)
