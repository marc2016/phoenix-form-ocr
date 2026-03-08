import cv2
import numpy as np
import base64
import pytesseract
from typing import List, Dict, Any, Tuple
from core.utils import id_generator, find_heading_above
from core.recognizers.checkbox import extract_checkbox, is_checkbox
from core.recognizers.textfield import extract_textfield, is_textfield
from core.recognizers.titlefield import extract_heading_for_checkboxes

def process_image_to_elements(image: np.ndarray) -> Tuple[List[Dict[str, Any]], str]:
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
    
    # Sort by bounding box area (smallest first) so checkboxes are processed before huge container table cells
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects.sort(key=lambda r: r[2] * r[3])
    
    elements = []
    annotated_image = image.copy()
    
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
                'text': data['text'][i],
                'id': i
            })

    added_boxes = []
    used_tbs_set = set()
    raw_fields = []

    for x, y, w, h in rects:
        
        # Filter for reasonable field sizes first so huge bounding boxes don't swallow everything
        # Evaluate standard sizes independently from recognition logic
        if not (is_checkbox(w, h) or is_textfield(w, h)):
            continue
            
        # Deduplicate overlapping boxes
        cx, cy = x + w/2, y + h/2
        is_duplicate = False
        for (ex, ey, ew, eh) in added_boxes:
            # 1. Check strict center proximity
            ecx, ecy = ex + ew/2, ey + eh/2
            if abs(cx - ecx) < 15 and abs(cy - ecy) < 15:
                is_duplicate = True
                break
                
            # 2. Check overlap (Intersection over Minimum Area)
            ix = max(x, ex)
            iy = max(y, ey)
            iw = min(x + w, ex + ew) - ix
            ih = min(y + h, ey + eh) - iy
            
            if iw > 0 and ih > 0:
                intersection_area = iw * ih
                min_area = min(w * h, ew * eh)
                # If one box is mostly containing or contained by the other, they are duplicates.
                if intersection_area > 0.4 * min_area:
                    is_duplicate = True
                    break
                    
        if is_duplicate:
            continue
            
        added_boxes.append((x, y, w, h))
        
        # Try extracting as Checkbox
        cb_field, cb_success = extract_checkbox(x, y, w, h, annotated_image, text_blocks, used_tbs_set)
        if cb_success:
            raw_fields.append(cb_field)
            continue
            
        # Try extracting as TextField
        tf_field, tf_success = extract_textfield(x, y, w, h, annotated_image, text_blocks, used_tbs_set)
        if tf_success:
            raw_fields.append(tf_field)
            continue
                
    # Sort fields by y then x
    raw_fields.sort(key=lambda f: (f['y'] // 20, f['x']))
    
    # Collect text fields normally
    for f in raw_fields:
        if f['type'] == 'TextField':
            elements.append({
                "y": f['y'],
                "x": f['x'],
                "element": {
                    "id": id_generator(),
                    "type": "TextField",
                    "extraAttributes": {
                        "label": f['label'],
                        "helperText": "",
                        "required": False,
                        "placeholder": f['placeholder']
                    }
                }
            })

    # Group checkboxes
    checkboxes = [f for f in raw_fields if f['type'] == 'CheckboxField']
    clusters = []
    
    for cb in checkboxes:
        added_to_cluster = False
        for cluster in clusters:
            # Check if this checkbox belongs to an existing cluster
            # A checkbox belongs if it's roughly on the same Y or close vertically/horizontally
            for member in cluster:
                # Same row (allow 25px variance) OR consecutive rows
                y_dist = abs(cb['y'] - member['y'])
                x_dist = abs(cb['x'] - member['x'])
                # Group if close horizontally on similar Y, or close vertically on similar X
                if (y_dist < 40 and x_dist < 300) or (x_dist < 50 and y_dist < 100):
                    cluster.append(cb)
                    added_to_cluster = True
                    break
            if added_to_cluster:
                break
        if not added_to_cluster:
            clusters.append([cb])
            
    # Process each cluster into a GroupField
    for cluster in clusters:
        cluster.sort(key=lambda f: (f['y'] // 20, f['x']))
        # Find bounds of the cluster
        min_y = min(f['y'] for f in cluster)
        min_x = min(f['x'] for f in cluster)
        max_y = max(f['y'] + f['h'] for f in cluster)
        max_x = max(f['x'] + f['w'] for f in cluster)
        
        # Check for heading above the entire cluster
        # Create a mock field representing the cluster for extract_heading_for_checkboxes
        mock_field = {
            'x': min_x, 'y': min_y, 'w': max_x - min_x, 'h': max_y - min_y
        }
        
        heading, heading_tbs = find_heading_above(mock_field['x'], mock_field['y'], mock_field['w'], mock_field['h'], text_blocks, used_tbs_set)
        
        if heading and len(heading) > 1:
            title = heading
            used_tbs_set.update(tb['id'] for tb in heading_tbs)
            
            for tb in heading_tbs:
                cv2.rectangle(annotated_image, (tb['x'], tb['y']), (tb['x']+tb['w'], tb['y']+tb['h']), (255, 0, 0), 2)
            cv2.putText(annotated_image, "HEADING", (min_x, min_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            title = "Checkbox Group"

        # Create GroupField
        group_content = []
        for cb in cluster:
            group_content.append({
                "id": id_generator(),
                "type": "CheckboxField",
                "extraAttributes": {
                    "label": cb['label'],
                    "helperText": "",
                    "required": False,
                    "checked": False
                }
            })
            
        elements.append({
            "y": min_y - 30, # Place the group slightly above the first checkbox
            "x": min_x,
            "element": {
                "id": id_generator(),
                "type": "GroupField",
                "extraAttributes": {
                    "title": title
                },
                "content": group_content
            }
        })
            
    # Sort final elements top to bottom, then left to right
    elements.sort(key=lambda e: (e['y'] // 20, e['x']))
    
    # Encode annotated image
    _, buffer = cv2.imencode('.jpg', annotated_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    return [e["element"] for e in elements], f"data:image/jpeg;base64,{base64_image}"
