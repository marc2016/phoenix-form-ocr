import sys
import os
sys.path.append("/app")

import cv2
import numpy as np
from core.pipeline import process_image_to_elements
from pdf2image import convert_from_path
import json

print("Converting pdf to image...")
images = convert_from_path("/app/test.pdf", dpi=200)

page1 = images[0]
open_cv_image = np.array(page1)[:, :, ::-1].copy()

import core.pipeline
import core.recognizers.checkbox

original_is_checkbox = core.recognizers.checkbox.is_checkbox

def debug_is_checkbox(w, h):
    res = original_is_checkbox(w, h)
    # Print near misses for checkbox
    if 15 < w < 80 and 15 < h < 80:
        print(f"Candidate: {w}x{h}, aspect={w/h:.2f} -> Checkbox Accepted: {res}")
    return res

core.recognizers.checkbox.is_checkbox = debug_is_checkbox

print("Processing...")
elements, _ = core.pipeline.process_image_to_elements(open_cv_image)

print("\n--- Extracted Elements ---")
for e in elements:
    print(f"Element: {e['type']} at {e.get('extraAttributes', {}).get('title', e.get('extraAttributes', {}).get('label', 'No Label'))}")
    if e["type"] == "GroupField":
        print(f"  Content ({len(e['content'])} fields):")
        for child in e["content"]:
            print(f"    - {child['type']}: {child['extraAttributes'].get('label')}")

