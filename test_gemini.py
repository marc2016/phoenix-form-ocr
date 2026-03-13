import os
import sys
import json
from core.ai_backend import GeminiBackend

def test_gemini(image_path):
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    print(f"Testing Gemini OCR with image: {image_path}")
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    backend = GeminiBackend()
    elements = backend.process_image(image_bytes)
    
    print(f"\nExtracted {len(elements)} elements:")
    print(json.dumps(elements, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_gemini.py <path_to_image>")
        sys.exit(1)
    
    test_gemini(sys.argv[1])
