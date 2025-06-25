from flask import Flask, request, jsonify
from PIL import Image
import easyocr
import numpy as np
import re
from io import BytesIO

app = Flask(__name__)
reader = easyocr.Reader(['en', 'hi'])  # English + Hindi OCR

def extract_dob_text(image: Image.Image):
    try:
        img_np = np.array(image)
        results = reader.readtext(img_np, detail=0, paragraph=True)
        text = " ".join(results)

        patterns = [
            r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",                                     # 12/05/1998
            r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",                                     # 1998-05-12
            r"\b\d{2}[/-]\d{2}[/-]\d{2}\b",                                     # 12/05/98
            r"\b\d{2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b", # 12 May 1998
            r"DOB[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})",
            r"Date of Birth[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})",
            r"जन्म[\s]*तिथि[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
    except Exception as e:
        print(f"[OCR ERROR]: {e}")
    return None

@app.route("/ocr", methods=["POST"])
def ocr_dob():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except:
        return jsonify({"error": "Invalid image format"}), 400

    dob_text = extract_dob_text(image)
    return jsonify({"dob_text": dob_text})

if __name__ == "__main__":
    app.run(port=8000)
