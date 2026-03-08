import os
import cv2
import numpy as np
from pdf2image import convert_from_path

def test_morphology(image: np.ndarray, page_num: int):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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
    
    contours, _ = cv2.findContours(table_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_img = image.copy()
    detected = 0
    added_boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Deduplicate
        cx, cy = x + w/2, y + h/2
        is_dup = False
        for (ex, ey, ew, eh) in added_boxes:
            ecx, ecy = ex + ew/2, ey + eh/2
            if abs(cx - ecx) < 15 and abs(cy - ecy) < 15:
                is_dup = True
                break
        if is_dup: continue
        
        added_boxes.append((x, y, w, h))
        
        is_checkbox = 25 < w < 50 and 25 < h < 50 and 0.7 < w/h < 1.3
        is_textfield = w > 100 and 20 < h < 100 and w/h > 2.0
        
        if is_checkbox:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            detected += 1
        elif is_textfield:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected += 1
            
    print(f"Page {page_num}: Found {detected} true geometric fields")
    cv2.imwrite(f"/Users/marclammers/.gemini/antigravity/brain/ada7963f-3fb8-4606-94b5-bc4219e51bb1/artifacts/morph_debug_page_{page_num}.jpg", debug_img)

if __name__ == "__main__":
    pdf_path = "mietzuschuss-nach-dem-wohngeldgesetz-bund-bf-v2.3.pdf"
    os.makedirs("/Users/marclammers/.gemini/antigravity/brain/ada7963f-3fb8-4606-94b5-bc4219e51bb1/artifacts", exist_ok=True)
    images = convert_from_path(pdf_path, dpi=200, first_page=2, last_page=2)
    img = np.array(images[0])[:, :, ::-1].copy()
    test_morphology(img, 2)
