import cv2
import numpy as np
import base64
import os
import pytesseract
from typing import List, Dict, Any, Tuple
from core.utils import id_generator, find_heading_above, clean_label
from core.recognizers.checkbox import extract_checkbox, is_checkbox
from core.recognizers.textfield import extract_textfield, is_textfield
from core.recognizers.titlefield import extract_heading_for_checkboxes
from core.ai_backend import GeminiBackend

def process_image_to_elements(image: np.ndarray, image_bytes: bytes = None) -> Tuple[List[Dict[str, Any]], str]:
    elements = []
    base64_image = None

    # Try Gemini Backend first if API Key is present
    if os.environ.get("GEMINI_API_KEY") and image_bytes:
        print("Using Gemini Backend for form recognition...")
        try:
            backend = GeminiBackend()
            elements = backend.process_image(image_bytes)
            
            if elements:
                print(f"Gemini found {len(elements)} elements. Proceeding with AI results.")
                annotated_image = image.copy()
                
                # 1. Draw bounding boxes on the image
                def draw_boxes(els):
                    for el in els:
                        box = el.get("box")
                        if box:
                            color = (0, 255, 0) if el["type"] == "TextField" else (0, 0, 255)
                            cv2.rectangle(annotated_image, (box['x'], box['y']), 
                                         (box['x'] + box['w'], box['y'] + box['h']), color, 2)
                        if el.get("type") == "GroupField" and "content" in el:
                            draw_boxes(el["content"])
                        elif el.get("type") == "TwoColumnField" and "columns" in el:
                            for col in el["columns"]:
                                draw_boxes(col)

                draw_boxes(elements)

                # 2. Clean up boxes
                def cleanup_boxes(els):
                    for el in els:
                        el.pop("box", None)
                        if el.get("type") == "GroupField" and "content" in el:
                            cleanup_boxes(el["content"])
                        elif el.get("type") == "TwoColumnField" and "columns" in el:
                            for col in el["columns"]:
                                cleanup_boxes(col)

                cleanup_boxes(elements)

                # 3. Encode
                _, buffer = cv2.imencode('.jpg', annotated_image)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                return elements, f"data:image/jpeg;base64,{base64_image}"
            else:
                print("Gemini returned no elements. Falling back to local OCR...")
        except Exception as e:
            print(f"Gemini backend failed: {e}. Falling back to local OCR...")

    # Fallback to Tesseract/OpenCV logic
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
    # 1. Identify all geometric boxes using hierarchy to find nested structures
    contours, hierarchy = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rects_with_hierarchy = []
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10: continue
            # parent_idx is hierarchy[0][i][3]
            rects_with_hierarchy.append({
                'rect': (x, y, w, h),
                'parent': hierarchy[0][i][3],
                'id': i
            })
    
    # Separate into Containers vs potential Leaf Fields
    # A container is a box that HAS children and is reasonably large
    containers = []
    leaf_candidates = []
    
    for item in rects_with_hierarchy:
        x, y, w, h = item['rect']
        
        # Check if it's a leaf (no children or children are too small to be fields)
        has_field_children = False
        for child in rects_with_hierarchy:
            if child['parent'] == item['id']:
                cw, ch = child['rect'][2], child['rect'][3]
                if is_checkbox(cw, ch) or is_textfield(cw, ch):
                    has_field_children = True
                    break
        
        if is_checkbox(w, h) or is_textfield(w, h):
            leaf_candidates.append(item['rect'])
        elif w > 40 and h > 20 and w < image.shape[1] * 0.9 and h < image.shape[0] * 0.9:
            # It's a container candidate if it's not a field but large enough
            containers.append(item['rect'])

    # 2. Process Leaf Fields (Checkboxes and TextFields)
    # Deduplicate leaves first
    unique_leaves = []
    for lx, ly, lw, lh in leaf_candidates:
        is_dup = False
        for ux, uy, uw, uh in unique_leaves:
            if abs(lx-ux) < 10 and abs(ly-uy) < 10:
                is_dup = True; break
        if not is_dup: unique_leaves.append((lx, ly, lw, lh))

    raw_fields = []
    for lx, ly, lw, lh in unique_leaves:
        if is_checkbox(lw, lh):
            raw_fields.append({"type": "CheckboxField", "x": lx, "y": ly, "w": lw, "h": lh, "label": "Checkbox"})
            cv2.rectangle(annotated_image, (lx, ly), (lx + lw, ly + lh), (0, 0, 255), 2)
        elif is_textfield(lw, lh):
            raw_fields.append({"type": "TextField", "x": lx, "y": ly, "w": lw, "h": lh, "label": "Text Field"})
            cv2.rectangle(annotated_image, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)

    # 3. Associate Fields with Containers
    # Sort boxes by area ASCENDING so we match the smallest (innermost) container first
    containers.sort(key=lambda c: c[2] * c[3])
    
    used_tbs_global = set()
    
    # Map fields to their smallest containing box
    container_map = {i: [] for i in range(len(containers))}
    orphan_fields = []

    for f_idx, f in enumerate(raw_fields):
        found_container = False
        for c_idx, (cx, cy, cw, ch) in enumerate(containers):
            # Strict containment
            if cx-2 <= f['x'] and cy-2 <= f['y'] and (f['x']+f['w']) <= (cx+cw+2) and (f['y']+f['h']) <= (cy+ch+2):
                container_map[c_idx].append(f)
                found_container = True
                break # Matched smallest container due to sort
        if not found_container:
            orphan_fields.append(f)
    
    # Process containers that actually have fields
    for c_idx, (cx, cy, cw, ch) in enumerate(containers):
        inner_fields = container_map[c_idx]
        if not inner_fields: continue

        # Find text inside this box
        inner_text_blocks = []
        for tb in text_blocks:
            if tb['id'] in used_tbs_global: continue
            if cx <= (tb['x'] + tb['w']/2) <= cx+cw and cy <= (tb['y'] + tb['h']/2) <= cy+ch:
                inner_text_blocks.append(tb)

        cv2.rectangle(annotated_image, (cx, cy), (cx+cw, cy+ch), (255, 0, 255), 2)

        # PASS 1: Assign text blocks to individual fields using STRICT rules
        used_tbs_in_container = set()
        inner_fields.sort(key=lambda f: (f['y'] // 15, f['x']))
        
        for f in inner_fields:
            best_score = 99999
            best_tb = None
            
            for tb in inner_text_blocks:
                if tb['id'] in used_tbs_in_container: continue
                
                # Check if text is INSIDE the field box
                # Robust margin for top (-15px)
                is_inside = (f['x']-5 <= tb['x'] and (tb['x']+tb['w']) <= f['x']+f['w']+5 and \
                            f['y']-15 <= tb['y'] and (tb['y']+tb['h']) <= f['y']+f['h']+5)
                
                if f['type'] == 'TextField':
                    # Rule: TextField labels are ALWAYS inside the box
                    if is_inside:
                        score = (tb['y'] - f['y']) + (tb['x'] - f['x'])
                    else: score = 99999
                else: # CheckboxField
                    # Rule: Checkbox labels are ALWAYS to the right
                    dy = abs(tb['y'] - f['y'])
                    dx_right = tb['x'] - (f['x'] + f['w'])
                    if dy < 20 and 0 <= dx_right < 300:
                        score = dx_right + (dy * 2)
                    else: score = 99999

                if score < best_score:
                    best_score = score
                    best_tb = tb

            if best_tb:
                # Expand label horizontally
                assigned_tbs = [best_tb]
                line_tbs = [tb for tb in inner_text_blocks if abs(tb['y'] - best_tb['y']) < 15 and tb['id'] not in used_tbs_in_container]
                line_tbs.sort(key=lambda b: b['x'])
                try: idx = line_tbs.index(best_tb)
                except: idx = -1
                if idx != -1:
                    l_ptr = idx
                    while l_ptr > 0:
                        if line_tbs[l_ptr]['x'] - (line_tbs[l_ptr-1]['x'] + line_tbs[l_ptr-1]['w']) < 60:
                            assigned_tbs.append(line_tbs[l_ptr-1]); l_ptr -= 1
                        else: break
                    r_ptr = idx
                    while r_ptr < len(line_tbs) - 1:
                        if line_tbs[r_ptr+1]['x'] - (line_tbs[r_ptr]['x'] + line_tbs[r_ptr]['w']) < 60:
                            assigned_tbs.append(line_tbs[r_ptr+1]); r_ptr += 1
                        else: break
                
                assigned_tbs.sort(key=lambda b: b['x'])
                f['label'] = clean_label(" ".join([tb['text'] for tb in assigned_tbs]))
                for tb in assigned_tbs:
                    used_tbs_in_container.add(tb['id'])
                    used_tbs_global.add(tb['id'])
            else:
                f['label'] = ""

        # PASS 2: Container Title
        # 2a. Look INSIDE (Top-most)
        remaining_tbs = [tb for tb in inner_text_blocks if tb['id'] not in used_tbs_in_container]
        filtered_tbs = [tb for tb in remaining_tbs if len(tb['text'].strip()) > 1 or tb['text'].strip() not in "[]()<>|_-.>/\\"]
        
        first_field_y = min(f['y'] for f in inner_fields)
        title_tbs = [tb for tb in filtered_tbs if tb['y'] < first_field_y + 25]
        title_tbs.sort(key=lambda b: (b['y'] // 10, b['x']))
        container_title = clean_label(" ".join([tb['text'] for tb in title_tbs]))
        
        # 2b. Fallback to OUTSIDE (Above the box)
        if not container_title:
            external_tbs = []
            for tb in text_blocks:
                if tb['id'] in used_tbs_global: continue
                # Closely above the box (max 80px)
                dist_above = cy - (tb['y'] + tb['h'])
                # Alignment check: must overlap horizontally OR be very close to left edge
                h_overlap = min(cx + cw, tb['x'] + tb['w']) - max(cx, tb['x'])
                if 0 <= dist_above < 80 and (h_overlap > 0 or abs(tb['x'] - cx) < 50):
                    external_tbs.append(tb)
            
            if external_tbs:
                # Only pick the closest line
                external_tbs.sort(key=lambda b: abs(cy - (b['y'] + b['h'])))
                closest_y = external_tbs[0]['y']
                line_tbs = [tb for tb in external_tbs if abs(tb['y'] - closest_y) < 15]
                line_tbs.sort(key=lambda b: b['x'])
                
                container_title = clean_label(" ".join([tb['text'] for tb in line_tbs]))
                used_tbs_global.update(tb['id'] for tb in line_tbs)

        if len(container_title) > 200: container_title = container_title[:197] + "..."
        used_tbs_global.update(tb['id'] for tb in inner_text_blocks)

        # Generate Element
        is_checkbox_group = all(f['type'] == 'CheckboxField' for f in inner_fields)
        
        if len(inner_fields) == 1:
            f = inner_fields[0]
            # If we have both a container title (heading) AND a field label, 
            # we should create a GroupField to keep them separate as per user feedback.
            if container_title and f['label'] and container_title.lower() != f['label'].lower():
                if f['type'] == 'CheckboxField':
                    elements.append({"y": cy, "x": cx, "element": {
                        "id": id_generator(), "type": "GroupField", 
                        "extraAttributes": {"title": container_title},
                        "content": [{"id": id_generator(), "type": "CheckboxField", "extraAttributes": {"label": f['label'], "helperText": "", "required": False, "checked": False}}]
                    }})
                else:
                    elements.append({"y": cy, "x": cx, "element": {
                        "id": id_generator(), "type": "GroupField", 
                        "extraAttributes": {"title": container_title},
                        "content": [{"id": id_generator(), "type": "TextField", "extraAttributes": {"label": f['label'], "helperText": "", "required": False, "placeholder": ""}}]
                    }})
            else:
                # Merge or pick the best one if they are similar or one is missing
                final_label = container_title or f['label'] or ("Option" if is_checkbox_group else "Eingabefeld")
                
                if f['type'] == 'CheckboxField':
                    elements.append({"y": cy, "x": cx, "element": {
                        "id": id_generator(), "type": "GroupField", 
                        "extraAttributes": {"title": final_label},
                        "content": [{"id": id_generator(), "type": "CheckboxField", "extraAttributes": {"label": f['label'] or "Ja", "helperText": "", "required": False, "checked": False}}]
                    }})
                else:
                    elements.append({"y": cy, "x": cx, "element": {
                        "id": id_generator(), "type": "TextField", 
                        "extraAttributes": {"label": final_label, "helperText": "", "required": False, "placeholder": ""}
                    }})
        else:
            content = []
            for f in inner_fields:
                if f['type'] == 'CheckboxField':
                    content.append({"id": id_generator(), "type": "CheckboxField", "extraAttributes": {"label": f['label'] or "Option", "helperText": "", "required": False, "checked": False}})
                else:
                    content.append({"id": id_generator(), "type": "TextField", "extraAttributes": {"label": f['label'] or "Eingabe", "helperText": "", "required": False, "placeholder": ""}})
            
            default_group_title = "Optionen" if is_checkbox_group else "Gruppe"
            elements.append({"y": cy, "x": cx, "element": {
                "id": id_generator(), "type": "GroupField", 
                "extraAttributes": {"title": container_title or default_group_title}, 
                "content": content
            }})

    # 4. Handle orphans (fields not contained in any box)
    for f in orphan_fields:
        # Use fallback proximity logic or just add as unlabeled
        elements.append({
            "y": f['y'], "x": f['x'],
            "element": {
                "id": id_generator(), "type": f['type'],
                "extraAttributes": {"label": f.get('label', "Unlabeled"), "helperText": "", "required": False, "placeholder": ""}
            }
        })

    # Sort final elements top to bottom
    elements.sort(key=lambda e: e['y'])
    
    grouped_rows = []
    if elements:
        current_row = [elements[0]]
        for i in range(1, len(elements)):
            # If within 25px Y-distance, same row
            if abs(elements[i]['y'] - current_row[0]['y']) < 25:
                current_row.append(elements[i])
            else:
                grouped_rows.append(current_row)
                current_row = [elements[i]]
        grouped_rows.append(current_row)

    final_output = []
    for row in grouped_rows:
        row.sort(key=lambda e: e['x'])
        if len(row) == 2:
            # Side-by-side pair
            final_output.append({
                "id": id_generator(),
                "type": "TwoColumnField",
                "extraAttributes": {},
                "columns": [[row[0]['element']], [row[1]['element']]]
            })
        elif len(row) > 2 and len(row) <= 4:
            # Potential multi-column row (e.g. checkboxes)
            # Check if all are checkboxes or small groups
            is_checkbox_row = all(
                e['element']['type'] == 'CheckboxField' or 
                (e['element']['type'] == 'GroupField' and len(e['element'].get('content', [])) <= 1)
                for e in row
            )
            if is_checkbox_row:
                content = []
                for e in row:
                    if e['element']['type'] == 'GroupField':
                        content.extend(e['element'].get('content', []))
                    else:
                        content.append(e['element'])
                
                final_output.append({
                    "id": id_generator(),
                    "type": "GroupField",
                    "extraAttributes": {"title": "", "gridColumns": len(row)},
                    "content": content
                })
            else:
                for e in row: final_output.append(e['element'])
        else:
            for e in row:
                final_output.append(e['element'])
    
    # Encode annotated image
    _, buffer = cv2.imencode('.jpg', annotated_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    return final_output, f"data:image/jpeg;base64,{base64_image}"
