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
    allow_credentials=True,
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
    
    # Find horizontal and vertical lines for box detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine lines to find boxes
    table_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    _, table_mask = cv2.threshold(table_lines, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours (potential input fields or checkboxes)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

    def find_nearest_label(x, y, max_dist=150):
        # Look for text to the left or above the box
        best_label = "Unlabeled Field"
        closest_dist = float('inf')
        
        for tb in text_blocks:
            # Check if text is to the left
            if tb['y'] > y - 20 and tb['y'] < y + 30 and tb['x'] < x:
                dist = x - (tb['x'] + tb['w'])
                if 0 < dist < closest_dist and dist < max_dist:
                    closest_dist = dist
                    best_label = tb['text']
            # Check if text is above
            elif tb['x'] > x - 20 and tb['x'] < x + 50 and tb['y'] < y:
                dist = y - (tb['y'] + tb['h'])
                if 0 < dist < closest_dist and dist < max_dist:
                    closest_dist = dist
                    best_label = tb['text']
                    
        return best_label

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter for reasonable field sizes
        # Checkboxes are usually small squares
        if 10 < w < 50 and 10 < h < 50 and 0.8 < w/h < 1.2:
            label = find_nearest_label(x, y)
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
        elif w > 50 and 10 < h < 100:
            label = find_nearest_label(x, y)
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
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Ensure the file is a PDF")
        
    try:
        contents = await file.read()
        
        # Convert PDF to images
        images = convert_from_bytes(contents, dpi=200)
        
        all_elements = []
        
        for i, pil_image in enumerate(images):
            # Convert PIL to OpenCV format
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            elements = process_image_to_elements(open_cv_image)
            all_elements.extend(elements)
            
        return {"elements": all_elements}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
