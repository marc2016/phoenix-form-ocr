import cv2
from typing import Dict, Any, List, Set, Tuple
from core.utils import id_generator, find_heading_above

def extract_heading_for_checkboxes(
    current_group_y: int, 
    f: Dict[str, Any], 
    text_blocks: List[Dict[str, Any]], 
    used_tbs_set: Set[int],
    annotated_image: Any
) -> Tuple[Dict[str, Any], int]:
    """
    Checks if a CheckboxField starts a new group. If so, looks for a heading above it.
    Returns the TitleField element (or None) and the new current_group_y.
    """
    if abs(f['y'] - current_group_y) > 25:
        # Find heading above this group
        heading, heading_tbs = find_heading_above(f['x'], f['y'], f['w'], f['h'], text_blocks, used_tbs_set)
        if heading and len(heading) > 1:
            heading_y = min((tb['y'] for tb in heading_tbs), default=f['y'] - 30)
            heading_x = min((tb['x'] for tb in heading_tbs), default=f['x'])
            used_tbs_set.update(tb['id'] for tb in heading_tbs)
            
            for tb in heading_tbs:
                cv2.rectangle(annotated_image, (tb['x'], tb['y']), (tb['x']+tb['w'], tb['y']+tb['h']), (255, 0, 0), 2)
            cv2.putText(annotated_image, "HEADING", (heading_x, heading_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            title_field = {
                "y": heading_y,
                "x": heading_x,
                "element": {
                    "id": id_generator(),
                    "type": "TitleField",
                    "extraAttributes": {
                        "title": heading
                    }
                }
            }
            return title_field, f['y']
        
        # Started a new row of checkboxes, but no valid heading found. 
        # Update current_group_y so we don't spam checking for this particular row.
        return None, f['y']
            
    # Same row of checkboxes as before, don't update current_group_y
    return None, current_group_y
