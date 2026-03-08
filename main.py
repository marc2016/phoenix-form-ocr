import os
import uuid
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from typing import List, Dict, Any

app = FastAPI(title="Phoenix Form OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def id_generator() -> str:
    return str(uuid.uuid4())

def process_image_to_elements(image: np.ndarray) -> List[Dict[str, Any]]:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to binarize
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # Use morphological operations to keep ONLY horizontal and vertical lines larger than 15 pixels
    # This destroys text but keeps the borders of checkboxes and text fields
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    
    # Connect broken borders slightly
    table_mask = cv2.dilate(table_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    # Find contours explicitly on geometric lines, not on text blobs
    contours, _ = cv2.findContours(table_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    elements = []
    
    # Also extract all text to find the nearest labels
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT, lang='eng+deu')
    
    text_blocks = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():
            text_blocks.append({
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i],
                'text': data['text'][i]
            })

    def find_nearest_label(x, y, w, h, is_checkbox):
        best_label = "Unlabeled Field"
        closest_dist = float('inf')
        
        # If textfield, often the label is INSIDE the box (top left)
        if not is_checkbox:
            inside_texts = []
            for tb in text_blocks:
                # Center of text block
                cx = tb['x'] + tb['w']/2
                cy = tb['y'] + tb['h']/2
                if x <= cx <= x + w and y <= cy <= y + h:
                    inside_texts.append(tb)
            if inside_texts:
                # sort by y then x
                inside_texts.sort(key=lambda b: (b['y'] // 10, b['x']))
                # Clean up specific label artifacts like '>'
                raw_label = " ".join([b['text'] for b in inside_texts])
                return raw_label.replace(" >", "").replace(">", "").strip()
                
        for tb in text_blocks:
            # Check if text is to the left
            if tb['y'] > y - 20 and tb['y'] < y + h + 20 and tb['x'] < x:
                dist = x - (tb['x'] + tb['w'])
                if 0 <= dist < closest_dist and dist < 200:
                    closest_dist = dist
                    best_label = tb['text']
            # Check if text is above
            elif tb['x'] > x - 50 and tb['x'] < x + w + 50 and tb['y'] < y:
                dist = y - (tb['y'] + tb['h'])
                if 0 <= dist < closest_dist and dist < 100:
                    closest_dist = dist
                    best_label = tb['text']
            # Check if text is to the right (useful for checkboxes)
            elif is_checkbox and tb['y'] > y - 20 and tb['y'] < y + h + 20 and tb['x'] > x + w:
                dist = tb['x'] - (x + w)
                if 0 <= dist < closest_dist and dist < 150:
                    closest_dist = dist
                    best_label = tb['text']
                    
        return best_label.replace(" >", "").replace(">", "").strip()

    added_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Deduplicate overlapping boxes (e.g. inner vs outer border of the same checkbox)
        cx, cy = x + w/2, y + h/2
        is_duplicate = False
        for (ex, ey, ew, eh) in added_boxes:
            ecx, ecy = ex + ew/2, ey + eh/2
            if abs(cx - ecx) < 15 and abs(cy - ecy) < 15:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
            
        added_boxes.append((x, y, w, h))
        
        # Filter for reasonable field sizes
        # Look for boxes that are likely checkboxes (small, square)
        # Bounding 33-45 perfectly isolates the standard checkbox size while avoiding thick letters
        is_checkbox = 33 < w < 45 and 33 < h < 45 and 0.8 < w/h < 1.2
        
        # Look for boxes that are likely text fields (long, not too tall)
        # Avoid false positives by ensuring width is significantly larger than height,
        # and height is reasonable for 1 line of text.
        is_textfield = w > 150 and 40 < h < 90 and w/h > 2.0
        
        def is_valid_label(l: str) -> bool:
            clean = l.strip()
            # Reject if it's the fallback
            if clean == "Unlabeled Field": return False
            # Reject if it's practically empty or just a single character (noise, 'O', 'x', etc)
            if len(clean) <= 1: return False
            # Reject common Tesseract noise strings
            if clean.lower() in ["oo", "==", "--", ">>", "~~"]: return False
            return True

        # Checkboxes are usually small squares
        if is_checkbox:
            label = find_nearest_label(x, y, w, h, True)
            if is_valid_label(label):
                elements.append({
                    "y": y,
                    "x": x,
                    "element": {
                        "id": id_generator(),
                        "type": "TextField", # Treating as TextField for now, or Checkbox if added to types
                        "extraAttributes": {
                            "label": f"{label} (Box)",
                            "helperText": "",
                            "required": False,
                            "placeHolder": ""
                        }
                    }
                })
        # Text fields are usually wider rectangles
        elif is_textfield:
            label = find_nearest_label(x, y, w, h, False)
            if is_valid_label(label):
                elements.append({
                    "y": y,
                    "x": x,
                    "element": {
                        "id": id_generator(),
                        "type": "TextField",
                        "extraAttributes": {
                            "label": label,
                            "helperText": "",
                            "required": False,
                            "placeHolder": ""
                        }
                    }
                })
            
    # Sort elements top to bottom, then left to right
    elements.sort(key=lambda e: (e['y'] // 20, e['x']))
    
    return [e["element"] for e in elements]

@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    print(f"Received request to parse-pdf. Filename: {file.filename}, Content-Type: {file.content_type}")
    if not file.filename.lower().endswith('.pdf'):
        print("File is not a PDF, returning 400 error.")
        raise HTTPException(status_code=400, detail="Ensure the file is a PDF")
        
    try:
        print("Reading file contents...")
        contents = await file.read()
        print(f"Finished reading {len(contents)} bytes.")
        
        # Convert PDF to images
        print("Converting PDF to images using pdf2image...")
        images = convert_from_bytes(contents, dpi=200)
        print(f"Successfully converted PDF into {len(images)} images.")
        
        all_elements = []
        
        for i, pil_image in enumerate(images):
            print(f"Processing image page {i+1}/{len(images)}...")
            # Convert PIL to OpenCV format
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            elements = process_image_to_elements(open_cv_image)
            print(f"Found {len(elements)} elements on page {i+1}.")
            all_elements.extend(elements)
            
        print(f"Extraction complete. Returning {len(all_elements)} total elements.")
        return {"elements": all_elements}
        
    except Exception as e:
        print(f"Exception caught during OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
