from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import easyocr
from datetime import datetime
import re

reader = easyocr.Reader(['en'])

mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face_embedding(image: Image.Image):
    try:
        face = mtcnn(image)
        if face is not None:
            emb = resnet(face.unsqueeze(0)).detach().numpy().flatten()
            return emb
    except:
        return None
    return None

def compare_faces(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2))

def extract_dob_text(image: Image.Image):
    try:
        results = reader.readtext(np.array(image))
        text = " ".join([item[1] for item in results])
        patterns = [
            r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
            r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",
            r"\b\d{2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b",
            r"DOB[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})",
            r"Date of Birth[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})",
            r"जन्म[\s]*तिथि[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
    except:
        pass
    return None

def parse_age_from_dob(dob_text):
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
        "%d %b %Y", "%d %B %Y", "%Y/%m/%d"
    ]
    for fmt in formats:
        try:
            dob = datetime.strptime(dob_text.strip(), fmt)
            today = datetime.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except:
            continue
    digits = ''.join(filter(str.isdigit, dob_text))
    if len(digits) == 4:
        try:
            dob = datetime.strptime(digits, "%Y")
            today = datetime.today()
            return today.year - dob.year
        except:
            pass
    return None
