import cv2
from typing import Dict, Any, List, Set, Tuple
from core.utils import find_nearest_label, is_valid_label, id_generator

def is_checkbox(w: int, h: int) -> bool:
    """Check if the bounding box matches typical checkbox dimensions."""
    return 24 < w < 50 and 24 < h < 50 and 0.8 < w/h < 1.25

def extract_checkbox(
    x: int, y: int, w: int, h: int, 
    annotated_image: Any, 
    text_blocks: List[Dict[str, Any]], 
    used_tbs_set: Set[int]
) -> Tuple[Dict[str, Any], bool]:
    """
    Attempts to extract a CheckboxField from a bounding box.
    Returns the raw field dict and a boolean indicating success.
    """
    if not is_checkbox(w, h):
        return None, False
        
    label, used_tbs = find_nearest_label(x, y, w, h, True, text_blocks, used_tbs_set)
    if is_valid_label(label):
        used_tbs_set.update(tb['id'] for tb in used_tbs)
        # Draw on annotated image
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red for Checkbox
        cv2.putText(annotated_image, f"CB {w}x{h}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return {
            "type": "CheckboxField",
            "x": x, "y": y, "w": w, "h": h,
            "label": label,
            "placeholder": ""
        }, True
        
    return None, False
