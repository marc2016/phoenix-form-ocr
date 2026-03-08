import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

def test_labels(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    thresh = cv2.medianBlur(thresh, 3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
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
                return " ".join([b['text'] for b in inside_texts])
                
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
                    
        return best_label

    found = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        is_checkbox = 33 < w < 45 and 33 < h < 45 and 0.8 < w/h < 1.2
        is_textfield = w > 150 and 40 < h < 90 and w/h > 2.0
        
        if is_checkbox:
            label = find_nearest_label(x, y, w, h, True)
            print(f"Checkbox at {x},{y} -> Label: {label}")
            found += 1
        elif is_textfield:
            label = find_nearest_label(x, y, w, h, False)
            print(f"TextField at {x},{y} -> Label: {label}")
            found += 1
    print(f"Total fields: {found}")

pdf_path = "mietzuschuss-nach-dem-wohngeldgesetz-bund-bf-v2.3.pdf"
print(f"Loading {pdf_path} (Page 2 only)")
images = convert_from_path(pdf_path, dpi=200, first_page=2, last_page=2)
open_cv_image = np.array(images[0])
open_cv_image = open_cv_image[:, :, ::-1].copy()
test_labels(open_cv_image)
