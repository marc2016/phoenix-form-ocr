import cv2
import numpy as np
import core.pipeline
from pdf2image import convert_from_path
import os

pdf_path = "/app/test.pdf"
print("Converting pdf to image...")
images = convert_from_path(pdf_path, dpi=200)
image = images[0]
open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

print("Processing...")
elements, _ = core.pipeline.process_image_to_elements(open_cv_image)

print("\n--- Extracted Elements ---")
for e in elements:
    title = e.get('extraAttributes', {}).get('title', e.get('extraAttributes', {}).get('label', 'No Label'))
    print(f"Element: {e['type']} - '{title}'")
    if e["type"] == "GroupField":
        print(f"  Content ({len(e['content'])} fields):")
        for child in e["content"]:
            print(f"    - {child['type']}: {child['extraAttributes'].get('label')}")
