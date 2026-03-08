import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

def process_and_draw(image: np.ndarray, page_num: int, output_dir: str):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
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
        best_label = "Unlabeled"
        closest_dist = float('inf')
        
        if not is_checkbox:
            inside_texts = []
            for tb in text_blocks:
                cx = tb['x'] + tb['w']/2
                cy = tb['y'] + tb['h']/2
                if x <= cx <= x + w and y <= cy <= y + h:
                    inside_texts.append(tb)
            if inside_texts:
                inside_texts.sort(key=lambda b: (b['y'] // 10, b['x']))
                raw_label = " ".join([b['text'] for b in inside_texts])
                return raw_label.replace(" >", "").replace(">", "").strip()
                
        for tb in text_blocks:
            if tb['y'] > y - 20 and tb['y'] < y + h + 20 and tb['x'] < x:
                dist = x - (tb['x'] + tb['w'])
                if 0 <= dist < closest_dist and dist < 200:
                    closest_dist = dist
                    best_label = tb['text']
            elif tb['x'] > x - 50 and tb['x'] < x + w + 50 and tb['y'] < y:
                dist = y - (tb['y'] + tb['h'])
                if 0 <= dist < closest_dist and dist < 100:
                    closest_dist = dist
                    best_label = tb['text']
            elif is_checkbox and tb['y'] > y - 20 and tb['y'] < y + h + 20 and tb['x'] > x + w:
                dist = tb['x'] - (x + w)
                if 0 <= dist < closest_dist and dist < 150:
                    closest_dist = dist
                    best_label = tb['text']
                    
        return best_label.replace(" >", "").replace(">", "").strip()

    debug_img = image.copy()
    detected_count = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        is_checkbox = 33 < w < 45 and 33 < h < 45 and 0.8 < w/h < 1.2
        is_textfield = w > 150 and 40 < h < 90 and w/h > 2.0
        
        if is_checkbox:
            label = find_nearest_label(x, y, w, h, True)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Encode label to avoid Unicode issues in cv2.putText
            disp_label = label.encode('ascii', 'ignore').decode('ascii')
            cv2.putText(debug_img, f"CB: {disp_label}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            detected_count += 1
        elif is_textfield:
            label = find_nearest_label(x, y, w, h, False)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            disp_label = label.encode('ascii', 'ignore').decode('ascii')
            cv2.putText(debug_img, f"TF: {disp_label}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)
            detected_count += 1
            
    print(f"Page {page_num}: Found {detected_count} fields.")
    
    output_path = os.path.join(output_dir, f"ocr_debug_page_{page_num}.jpg")
    cv2.imwrite(output_path, debug_img)

def process_all_pages():
    pdf_path = "mietzuschuss-nach-dem-wohngeldgesetz-bund-bf-v2.3.pdf"
    output_dir = "/Users/marclammers/.gemini/antigravity/brain/ada7963f-3fb8-4606-94b5-bc4219e51bb1/artifacts/"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading FULL PDF: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=200)
    print(f"Loaded {len(images)} pages. Processing...")
    
    for i, pil_img in enumerate(images):
        page_num = i + 1
        open_cv_image = np.array(pil_img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        process_and_draw(open_cv_image, page_num, output_dir)
        
    print(f"All {len(images)} pages processed and saved to {output_dir}")

if __name__ == "__main__":
    process_all_pages()
