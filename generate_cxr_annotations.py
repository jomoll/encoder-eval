import pandas as pd
import os
import json
import asyncio
import aiohttp
from datasets import load_dataset, Dataset
from google import genai
from google.genai import types as genai_types
from tqdm.asyncio import tqdm

CXR_IMAMGES = "data/mimic-cxr-images-512"
API_KEY = "AIzaSyC-LwOBfcEiBmrVHwYxYW0NVzUdoC1GEqM"
PROMPT = """You are labeling a chest radiograph CORNER CROP. 
Task: decide if a true laterality marker is present in THIS CROP only.

Laterality markers are single letters "L" or "R" (or the words LEFT/RIGHT) physically burned into the image or as small digital overlays near the border.

IGNORE and do NOT count: AP, PA, LAT, SUPINE, PORTABLE, TECHNICAL OVERLAYS, DATES, PATIENT NAMES, HOSPITAL NAMES, GRIDLINES, WATERMARKS.

Return ONLY strict JSON:
{{
  "marker_present": <true|false>,
  "marker": "<L|R|UNKNOWN>",
  "bbox": [x, y, w, h],  // pixel coords in this crop, or [] if absent
  "confidence": <0.0-1.0>
}}"""

# Create async client
client = genai.Client(api_key=API_KEY)

async def gen_response_async(image, semaphore):
    """Generate response with rate limiting"""
    async with semaphore:  # Limit concurrent requests
        try:
            cfg = genai_types.GenerateContentConfig(response_mime_type="application/json")
            # Note: If genai client doesn't support async, use aiohttp instead
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-pro", 
                contents=[image, PROMPT],
                config=cfg
            )
            return response
        except Exception as e:
            print(f"API error: {e}")
            return None

def parse_response(response):
    """Parse the JSON response"""
    if not response:
        return None
    try:
        data = json.loads(response.text)
        marker_present = data.get("marker_present", False)
        marker = data.get("marker", "UNKNOWN")
        bbox = data.get("bbox", [])
        confidence = data.get("confidence", 0.0)
        return {
            "marker_present": marker_present,
            "marker": marker,
            "bbox": bbox,
            "confidence": confidence
        }
    except json.JSONDecodeError:
        print("Failed to parse JSON")
        return None

async def process_single_item(item, semaphore):
    """Process a single dataset item"""
    try:
        image = item['image']
        response = await gen_response_async(image, semaphore)
        data = parse_response(response)
        
        if data:
            return {
                "marker_present": data["marker_present"], 
                "marker": data["marker"],
                "bbox": data["bbox"],
                "confidence": data["confidence"]
            }
        else:
            return {
                "marker_present": False,
                "marker": "UNKNOWN", 
                "bbox": [],
                "confidence": 0.0
            }
    except Exception as e:
        print(f"Error processing {item.get('image_id', 'unknown')}: {e}")
        return None

async def generate_dataset_async(dataset, max_items=100, max_concurrent=10):
    """Generate dataset with async processing"""
    # Limit concurrent API calls to avoid rate limits
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Take subset of dataset
    items = dataset.select(range(min(len(dataset), max_items)))
    
    # Create tasks for all items
    tasks = [process_single_item(item, semaphore) for item in items]
    
    # Process with progress bar
    print(f"Processing {len(tasks)} images with {max_concurrent} concurrent requests...")
    results = await tqdm.gather(*tasks, desc="Processing images")
    
    # Filter out None results
    records = [r for r in results if r is not None]
    
    print(f"Successfully processed {len(records)}/{len(tasks)} images")
    
    # Create dataset
    new_df = pd.DataFrame(records)
    new_dataset = Dataset.from_pandas(new_df)
    # save locally
    new_dataset.save_to_disk("/data/moll/mimic-cxr-laterality-markers")
    # Upload to huggingface
    print("Uploading to Hugging Face...")
    new_dataset.push_to_hub("jomoll/mimic-cxr-laterality-markers")
    
    return new_dataset

def main():
    """Main function to run async processing"""
    dataset = load_dataset(CXR_IMAMGES)
    
    # Show first image for verification
    dataset["train"][0]['image'].show()
    
    # Run async processing
    result = asyncio.run(generate_dataset_async(
        dataset["train"], 
        max_items=100,
        max_concurrent=5  # Start conservative to avoid rate limits
    ))
    
    print(f"Generated dataset with {len(result)} annotations")
    return result

if __name__ == "__main__":
    main()
