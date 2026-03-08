import cv2
from typing import Dict, Any, List, Set, Tuple
from core.utils import find_nearest_label, is_valid_label, id_generator

def is_textfield(w: int, h: int) -> bool:
    """Check if bounding box matches typical text field dimensions."""
    return w > 150 and 40 < h < 90 and w/h > 2.0

def extract_textfield(
    x: int, y: int, w: int, h: int, 
    annotated_image: Any, 
    text_blocks: List[Dict[str, Any]], 
    used_tbs_set: Set[int]
) -> Tuple[Dict[str, Any], bool]:
    """
    Attempts to extract a TextField from a bounding box.
    Returns the raw field dict and a boolean indicating success.
    """
    if not is_textfield(w, h):
        return None, False
        
    label, used_tbs = find_nearest_label(x, y, w, h, False, text_blocks, used_tbs_set)
    if is_valid_label(label):
        used_tbs_set.update(tb['id'] for tb in used_tbs)
        # Draw on annotated image
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green for TextField
        cv2.putText(annotated_image, f"TF {w}x{h}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return {
            "type": "TextField",
            "x": x, "y": y, "w": w, "h": h,
            "label": label,
            "placeholder": f"Bitte {label} eingeben..."
        }, True
        
    return None, False
