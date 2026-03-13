import cv2
import numpy as np
import base64
import io
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
        print(f"--- Starting OCR Process for {file.filename} ---")
        print(f"Reading file contents (Size: {file.size if hasattr(file, 'size') else 'unknown'} bytes)...")
        contents = await file.read()
        file_size = len(contents)
        print(f"Read {file_size} bytes. Starting conversion...")
        
        if file_size == 0:
            print("ERROR: File contents are empty!")
            raise HTTPException(status_code=400, detail="Empty PDF file")

        # Convert PDF to images
        print(f"PDF to Image: calling pdf2image.convert_from_bytes for {file.filename}...")
        try:
            images = convert_from_bytes(contents, dpi=200)
            print(f"PDF to Image: Successfully converted into {len(images)} pages.")
        except Exception as pdf_err:
            print(f"CRITICAL ERROR in pdf2image: {str(pdf_err)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"pdf2image failed: {str(pdf_err)}")
        
        all_elements = []
        all_images = []
        
        for i, pil_image in enumerate(images):
            print(f"Page {i+1}/{len(images)}: Processing image ({pil_image.width}x{pil_image.height})...")
            # Convert PIL to OpenCV format
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR 
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            # Prepare bytes for Gemini
            print(f"Page {i+1}: Saving image as JPEG for analysis...")
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
            
            print(f"Page {i+1}: Calling process_image_to_elements (Pipeline)...")
            try:
                elements, base64_image = process_image_to_elements(open_cv_image, image_bytes=image_bytes)
                print(f"Page {i+1}: Found {len(elements)} elements.")
                all_elements.extend(elements)
                all_images.append(base64_image)
            except Exception as pipeline_err:
                print(f"ERROR on Page {i+1} during pipeline: {str(pipeline_err)}")
                # Continue or fail? Let's fail for now to be clear.
                raise pipeline_err
            
        print(f"--- Extraction Complete ---")
        print(f"Total elements found: {len(all_elements)}")
        return {"elements": all_elements, "images": all_images}
        
    except Exception as e:
        print(f"FATAL UI ERROR during OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return 500 with detailed error message so frontend can show it
        raise HTTPException(status_code=500, detail=f"OCR Server Error: {str(e)}")
