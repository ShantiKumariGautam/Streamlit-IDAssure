import streamlit as st
from PIL import Image
from datetime import datetime
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import easyocr
import re
import fitz

st.set_page_config(page_title="IDAssure", layout="centered")
st.markdown("<meta http-equiv='refresh' content='540'>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.6em 2em;
        border-radius: 10px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #4CAF50;
        background-color:#4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

reader = get_ocr_reader()

@st.cache_resource
def load_face_models():
    mtcnn = MTCNN(image_size=160, margin=20)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

mtcnn, resnet = load_face_models()

def extract_face_embedding(image: Image.Image):
    face = mtcnn(image)
    if face is not None:
        return resnet(face.unsqueeze(0)).detach().numpy().flatten()
    return None

def compare_faces(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2))

@st.cache_data
def extract_image_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    return None

def extract_dob_text(image: Image.Image):
    results = reader.readtext(np.array(image))
    text = " ".join([item[1] for item in results])
    patterns = [
        r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        r"\b\d{2}[/-]\d{2}[/-]\d{2}\b",
        r"\b\d{2}\s+\w+\s+\d{4}\b",
        r"\b\d{1,2}[ ]?[A-Za-z]{3,9}[ ]?\d{4}\b",
        r"DOB[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})",
        r"Date of Birth[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})",
        r"Birth[:\s\-]*([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None

def parse_age_from_dob(dob_text):
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %B %Y", "%d %b %Y", "%d%m%Y",
        "%Y", "%d %m %Y"
    ]
    for fmt in formats:
        try:
            dob = datetime.strptime(dob_text.strip(), fmt)
            today = datetime.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except:
            continue
    return None

st.title("IDAssure : Smart Identity Verification Portal")

aadhar_file = st.file_uploader("Upload Identity Card (Image or PDF)", type=["jpg", "jpeg", "png", "pdf"])
selfie_file = st.camera_input("Take your selfie")

with st.expander("Selfie Guidelines"):
    st.markdown("""
    - 🔆 **Take selfie in a well-lit area**  
    - 👁️ **Keep your eyes open and look at the camera**  
    - 😊 **Face should be fully visible (no masks or sunglasses)**  
    - 📱 **Hold the camera steady to avoid blur**
    """)

st.markdown('<div class="center-button">', unsafe_allow_html=True)
submit = st.button("Verify Identity")
st.markdown('</div>', unsafe_allow_html=True)

if submit:
    if not aadhar_file or not selfie_file:
        st.warning("Please upload both Identity Card and Selfie")
    else:
        with st.spinner("Processing..."):
            # Load ID image
            if aadhar_file.type == "application/pdf":
                aadhar_img = extract_image_from_pdf(aadhar_file.getvalue())
            else:
                aadhar_img = Image.open(aadhar_file).convert("RGB")
            selfie_img = Image.open(selfie_file).convert("RGB")

            # Face embeddings
            emb1 = extract_face_embedding(aadhar_img)
            emb2 = extract_face_embedding(selfie_img)

            if emb1 is None or emb2 is None:
                st.error("❌ Face not detected properly in one or both images.")
            else:
                score = compare_faces(emb1, emb2)

                # DOB + Age
                dob_text = extract_dob_text(aadhar_img)
                age = parse_age_from_dob(dob_text) if dob_text else None

                # Output
                st.subheader("Results")
                st.write(f"Face Match: {score*100:.2f}%")
                st.write(f"DOB Text: {dob_text if dob_text else 'Not found'}")
                st.write(f"Estimated Age: {age if age else 'Not found'}")

                if not dob_text:
                    st.warning("DOB not found in the uploaded Identity Card.")

                # Final Decision
                if score > 0.70 and age and age >= 18:
                    st.success("✅ Identity and Age Verified")
                elif score > 0.70 and age is not None:
                    st.warning("Identity Verified, Age < 18")
                elif score > 0.70 and age is None:
                    st.warning("Face Matched, but DOB not found – Cannot verify age")
                elif score > 0.5:
                    st.warning("⚠️ Face match is low")
                else:
                    st.error("❌ Verification Failed. Try again with clearer images.")
