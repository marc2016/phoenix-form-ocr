import os
import uuid
import cv2
import numpy as np
import base64
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from typing import List, Dict, Any, Tuple

from core.pipeline import process_image_to_elements

app = FastAPI(title="Phoenix Form OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        all_images = []
        
        for i, pil_image in enumerate(images):
            print(f"Processing image page {i+1}/{len(images)}...")
            # Convert PIL to OpenCV format
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            elements, base64_image = process_image_to_elements(open_cv_image)
            print(f"Found {len(elements)} elements on page {i+1}.")
            all_elements.extend(elements)
            all_images.append(base64_image)
            
        print(f"Extraction complete. Returning {len(all_elements)} total elements.")
        return {"elements": all_elements, "images": all_images}
        
    except Exception as e:
        print(f"Exception caught during OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
