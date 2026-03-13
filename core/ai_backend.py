import os
from google import genai
from google.genai import types
import PIL.Image
import json
import base64
import io
from typing import List, Dict, Any, Tuple
from core.utils import id_generator

class GeminiBackend:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            # New SDK client initialization
            self.client = genai.Client(api_key=self.api_key)
            # Using exactly what the user provided in their snippet
            self.model_id = "gemini-3-flash-preview" 
        else:
            self.client = None

    def process_image(self, image_data: bytes) -> List[Dict[str, Any]]:
        if not self.client:
            print("Gemini client not initialized. Check GEMINI_API_KEY.")
            return []

        # Convert bytes to PIL Image for the SDK
        img = PIL.Image.open(io.BytesIO(image_data))
        width, height = img.size
        
        prompt = """
        Analyze this form image and extract ALL input fields, checkboxes, and their labels.
        
        CRITICAL: 
        1. Look for groups like 'Geschlecht' (männlich, weiblich, divers) or 'Familienstand' (ledig, verheiratet, etc.).
        2. Distinguish SECTION HEADLINES (e.g., "2 Wie lautet die Anschrift der Wohnung") from FIELD LABELS (e.g., "Straße"). 
        3. A Section Headline should ALWAYS be a 'GroupField' title. 
        4. Field Labels are the specific labels for a TextField or CheckboxField.
        5. Every single checkbox must be captured. 
        6. For every field, provide a bounding box 'box_2d' in [ymin, xmin, ymax, xmax] format (normalized 0-1000).
        7. If fields are positioned SIDE-BY-SIDE (horizontal), use a 'TwoColumnField'.
        
        FIM ALIGNMENT (Föderales Informationsmanagement):
        For each field, attempt to identify a 'fimType' or 'semanticType' (e.g., 'person.name.nachname', 'person.geburtsdatum', 'anschrift.strasse').
        If a clear FIM mapping is found, include it in the JSON as 'fimType'.
        
        Return the result ONLY as a VALID JSON object with an 'elements' key.
        JSON Structure:
        - For TextField: {"type": "TextField", "label": "...", "box_2d": [ymin, xmin, ymax, xmax], "fimType": "..."}
        - For CheckboxField: {"type": "CheckboxField", "label": "...", "box_2d": [ymin, xmin, ymax, xmax], "fimType": "..."}
        - For GroupField: {"type": "GroupField", "title": "...", "gridColumns": 1, "box_2d": [ymin, xmin, ymax, xmax], "content": [list of elements], "fimType": "..."}
        - For TwoColumnField: {"type": "TwoColumnField", "box_2d": [ymin, xmin, ymax, xmax], "columns": [[left_elements], [right_elements]]}
        
        Note: For Groups containing only Checkboxes, set "gridColumns": 4.
        
        IMPORTANT: Use proper JSON escaping. No trailing commas. No comments.
        """

        import time

        models_to_try = [
            "models/gemini-2.0-flash", 
            "models/gemini-flash-latest"
        ]
        
        for model_id in models_to_try:
            print(f"Attempting recognition with model: {model_id}")
            for attempt in range(3):
                try:
                    start_time = time.time()
                    print(f"Requesting Gemini API (Model: {model_id}, Attempt: {attempt + 1})...")
                    # Use the new SDK's generate_content method
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(
                            response_mime_type='application/json'
                        )
                    )
                    end_time = time.time()
                    print(f"Gemini API responded in {end_time - start_time:.2f}s")
                    
                    # The new SDK parses JSON automatically if response_mime_type is set
                    if hasattr(response, 'parsed') and response.parsed:
                        raw_elements = response.parsed.get("elements", [])
                        print(f"Successfully parsed elements from 'response.parsed' ({len(raw_elements)} elements)")
                    else:
                        # Fallback to text parsing if needed
                        text = response.text.strip()
                        print(f"Raw Gemini text response: {text[:200]}...")
                        data = json.loads(text)
                        raw_elements = data.get("elements", [])
                        print(f"Parsed {len(raw_elements)} elements from text response.")
                    
                    if raw_elements:
                        return self._transform_to_phoenix_format(raw_elements, width, height)
                    
                    # If we got response but no elements, maybe the model didn't find anything
                    # No need to retry this specific model, move to next model
                    print(f"Model {model_id} returned no elements.")
                    break

                except Exception as e:
                    print(f"Attempt {attempt + 1} for model {model_id} failed: {e}")
                    if "503" in str(e) or "429" in str(e):
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        break # Other errors might not be worth retrying
            
        return []

    def _transform_to_phoenix_format(self, raw_elements: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
        formatted = []
        
        def denormalize(box):
            if not box or len(box) != 4: return None
            ymin, xmin, ymax, xmax = box
            return {
                "x": int(xmin * img_w / 1000),
                "y": int(ymin * img_h / 1000),
                "w": int((xmax - xmin) * img_w / 1000),
                "h": int((ymax - ymin) * img_h / 1000)
            }

        def transform_element(el):
            el_type = el.get("type")
            box = denormalize(el.get("box_2d"))
            fim_type = el.get("fimType")
            
            if el_type == "TextField":
                item = {
                    "id": id_generator(),
                    "type": "TextField",
                    "extraAttributes": {
                        "label": el.get("label") or el.get("title") or "Eingabefeld",
                        "helperText": el.get("helperText", ""),
                        "required": el.get("required", False),
                        "placeholder": el.get("placeholder", ""),
                        "fimType": fim_type
                    }
                }
                if box: item["box"] = box
                return item
                
            elif el_type == "CheckboxField":
                 # In phoenix-form, a single checkbox is often wrapped in a group for consistency
                 item = {
                    "id": id_generator(),
                    "type": "GroupField",
                    "extraAttributes": {
                        "title": el.get("label") or "Option",
                        "fimType": fim_type
                    },
                    "content": [{
                        "id": id_generator(),
                        "type": "CheckboxField",
                        "extraAttributes": {
                            "label": el.get("label") or "Ja",
                            "helperText": "",
                            "required": False,
                            "checked": el.get("checked", False)
                        }
                    }]
                }
                 if box: item["box"] = box
                 return item

            elif el_type == "GroupField":
                content = []
                for child in el.get("content", []):
                    transformed = transform_element(child)
                    if transformed: content.append(transformed)
                
                item = {
                    "id": id_generator(),
                    "type": "GroupField",
                    "extraAttributes": {
                        "title": el.get("title") or "Gruppe",
                        "gridColumns": el.get("gridColumns") or 1,
                        "fimType": fim_type
                    },
                    "content": content
                }
                if box: item["box"] = box
                return item

            elif el_type == "TwoColumnField":
                col1 = []
                col2 = []
                raw_cols = el.get("columns", [[], []])
                for child in raw_cols[0]:
                    transformed = transform_element(child)
                    if transformed: col1.append(transformed)
                for child in raw_cols[1]:
                    transformed = transform_element(child)
                    if transformed: col2.append(transformed)
                
                item = {
                    "id": id_generator(),
                    "type": "TwoColumnField",
                    "extraAttributes": {},
                    "columns": [col1, col2]
                }
                if box: item["box"] = box
                return item
            
            return None

        for el in raw_elements:
            transformed = transform_element(el)
            if transformed:
                formatted.append(transformed)
        return formatted

if __name__ == "__main__":
    # Simple test if run directly
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "rb") as f:
            data = f.read()
        backend = GeminiBackend()
        results = backend.process_image(data)
        print(json.dumps(results, indent=2))
    else:
        print("Usage: python core/ai_backend.py <image_path>")
