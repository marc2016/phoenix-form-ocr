import uuid
import re

def clean_label(text: str) -> str:
    """Removes common Tesseract checkbox artifacts like [[] , L] , oO etc."""
    if not text: return ""
    
    # 1. Strip common multi-character noise at the start that often comes from checkbox borders
    noise_patterns = [
        r'^\[+\]*', r'^L\]', r'^J\]', r'^\[J', r'^\[L', r'^oO', r'^Oo', r'^LJ', r'^\[_\]', r'^\[ \]', r'^\[x\]', r'^\[X\]', 
        r'^[\|Il1]\s', r'^[OUuJj]\s', r'^[a-z]\s' # Single character noise followed by space
    ]
    
    current = text.strip()
    for pattern in noise_patterns:
        current = re.sub(pattern, '', current).strip()
    
    # 2. Strip standard bracket/separator/noise characters from the very beginning and end
    # Removed ( and ) from here to preserve labels like Vorname(n)
    current = current.lstrip('[]{}||-_>.<:  ')
    current = current.rstrip('[]{}||-_>.<:  ')
    
    return current

def id_generator() -> str:
    return str(uuid.uuid4())

def is_valid_label(l: str) -> bool:
    clean = l.strip()
    # Reject if it's the fallback
    if clean == "Unlabeled Field": return False
    # Reject if it's practically empty or just a single character (noise, 'O', 'x', etc)
    if len(clean) <= 1: return False
    # Reject common Tesseract noise strings
    if clean.lower() in ["oo", "==", "--", ">>", "~~"]: return False
    return True

def find_nearest_label(x, y, w, h, is_checkbox, text_blocks, used_tbs_set):
    best_tb = None
    closest_dist = float('inf')
    
    # If textfield, often the label is INSIDE the box (top left)
    if not is_checkbox:
        inside_texts = []
        for tb in text_blocks:
            if tb['id'] in used_tbs_set: continue
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
            return clean_label(raw_label), inside_texts
            
    # First pass: try to find label on the right (most common for checkboxes)
    if is_checkbox:
        for tb in text_blocks:
            if tb['id'] in used_tbs_set: continue
            if tb['y'] > y - 20 and tb['y'] < y + h + 20 and tb['x'] > x + w:
                dist = tb['x'] - (x + w)
                if 0 <= dist < closest_dist and dist < 150:
                    closest_dist = dist
                    best_tb = tb

    if best_tb is None:
        for tb in text_blocks:
            if tb['id'] in used_tbs_set: continue
            # Check if text is to the left
            if tb['y'] > y - 20 and tb['y'] < y + h + 20 and tb['x'] < x:
                dist = x - (tb['x'] + tb['w'])
                if 0 <= dist < closest_dist and dist < 200:
                    closest_dist = dist
                    best_tb = tb
            # Check if text is above
            elif tb['x'] > x - 50 and tb['x'] < x + w + 50 and tb['y'] < y:
                dist = y - (tb['y'] + tb['h'])
                # Checkboxes rarely have labels above (usually those are headings)
                limit = 40 if is_checkbox else 100
                if 0 <= dist < closest_dist and dist < limit:
                    closest_dist = dist
                    best_tb = tb
                
    if best_tb is None:
        return "Unlabeled Field", []
        
    # Group text blocks that are on the same line as best_tb
    line_tbs = []
    for tb in text_blocks:
        # Same line? (Y difference < 15)
        if abs(tb['y'] - best_tb['y']) < 15:
            line_tbs.append(tb)
            
    line_tbs.sort(key=lambda b: b['x'])
    
    try:
        best_idx = line_tbs.index(best_tb)
    except ValueError:
        best_idx = -1
        
    if best_idx != -1:
        # expand left
        left_idx = best_idx
        while left_idx > 0:
            dist_to_next = line_tbs[left_idx]['x'] - (line_tbs[left_idx-1]['x'] + line_tbs[left_idx-1]['w'])
            if dist_to_next > 60: # max horizontal distance between words to consider them same label
                break
            left_idx -= 1
            
        # expand right
        right_idx = best_idx
        while right_idx < len(line_tbs) - 1:
            dist_to_next = line_tbs[right_idx+1]['x'] - (line_tbs[right_idx]['x'] + line_tbs[right_idx]['w'])
            if dist_to_next > 60:
                break
            right_idx += 1
            
        line_tbs = line_tbs[left_idx:right_idx+1]
        
    raw_label = " ".join([b['text'] for b in line_tbs])
    return clean_label(raw_label), line_tbs

def find_heading_above(cx, cy, cw, ch, text_blocks, used_tbs_set):
    best_tb = None
    closest_dist = float('inf')
    for tb in text_blocks:
        if tb['id'] in used_tbs_set: continue
        if (tb['y'] + tb['h']) < cy:
            hoffset = 0
            if tb['x'] + tb['w'] < cx:
                hoffset = cx - (tb['x'] + tb['w'])
            elif tb['x'] > cx + cw:
                hoffset = tb['x'] - (cx + cw)
                
            if hoffset < 400:
                dist = cy - (tb['y'] + tb['h'])
                if 0 < dist < 150 and dist < closest_dist:
                    closest_dist = dist
                    best_tb = tb
                    
    if best_tb is None:
        return None, []
        
    line_tbs = []
    for tb in text_blocks:
        if abs(tb['y'] - best_tb['y']) < 15:
            line_tbs.append(tb)
    line_tbs.sort(key=lambda b: b['x'])
    
    try: best_idx = line_tbs.index(best_tb)
    except ValueError: best_idx = -1
        
    if best_idx != -1:
        left_idx = best_idx
        while left_idx > 0:
            dist_to_next = line_tbs[left_idx]['x'] - (line_tbs[left_idx-1]['x'] + line_tbs[left_idx-1]['w'])
            if dist_to_next > 60: break
            left_idx -= 1
        right_idx = best_idx
        while right_idx < len(line_tbs) - 1:
            dist_to_next = line_tbs[right_idx+1]['x'] - (line_tbs[right_idx]['x'] + line_tbs[right_idx]['w'])
            if dist_to_next > 60: break
            right_idx += 1
        line_tbs = line_tbs[left_idx:right_idx+1]
        
    raw_label = " ".join([b['text'] for b in line_tbs])
    return clean_label(raw_label), line_tbs
