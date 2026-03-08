import os
import cv2
import numpy as np
from pdf2image import convert_from_path

def process_image_to_elements_debug(image: np.ndarray, output_path: str):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to binarize
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # Optional: slight blur to connect broken lines
    thresh = cv2.medianBlur(thresh, 3)
    
    # Find all contours directly, rather than only perfect intersecting lines
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_img = image.copy()
    
    detected_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Current logic from main.py
        is_checkbox = 25 < w < 45 and 25 < h < 45 and 0.8 < w/h < 1.2
        is_textfield = w > 150 and 40 < h < 90 and w/h > 2.0
        
        # DEBUG: Draw all rectangles in blue to see what OpenCV is picking up
        # cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # print(f"Found rect: w={w}, h={h}")
        
        if is_checkbox:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red for Checkbox
            cv2.putText(debug_img, f"CB {w}x{h}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            detected_count += 1
        elif is_textfield:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green for TextField
            cv2.putText(debug_img, f"TF {w}x{h}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            detected_count += 1
            
    print(f"Total detected fields on page: {detected_count}")
    cv2.imwrite(output_path, debug_img)

pdf_path = "mietzuschuss-nach-dem-wohngeldgesetz-bund-bf-v2.3.pdf"
print(f"Loading {pdf_path} (Page 2 only)")
images = convert_from_path(pdf_path, dpi=200, first_page=2, last_page=2)

open_cv_image = np.array(images[0])
open_cv_image = open_cv_image[:, :, ::-1].copy()

output_path = "/Users/marclammers/.gemini/antigravity/brain/ada7963f-3fb8-4606-94b5-bc4219e51bb1/artifacts/ocr_debug_page2.jpg"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
process_image_to_elements_debug(open_cv_image, output_path)
print(f"Saved debug image to {output_path}")
