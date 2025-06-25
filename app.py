import streamlit as st
from PIL import Image
from datetime import datetime
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import easyocr
import re
import fitz  # PyMuPDF

# Models
mtcnn = MTCNN(image_size=160, margin=20, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().cpu()

# -------------------------- Utility Functions -------------------------- #

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

def extract_image_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    return None

def extract_dob_text(image: Image.Image):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(np.array(image))
    text = " ".join([item[1] for item in results])
    patterns = [
        r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        r"\b\d{2}[/-]\d{2}[/-]\d{2}\b",
        r"\b\d{2}\s+\w+\s+\d{4}\b",
        r"\b\d{1,2}[ ]?[A-Za-z]{3,9}[ ]?\d{4}\b",
        r"\b\d{4}\b",
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

# -------------------------- UI & App Logic -------------------------- #

st.set_page_config(page_title="IDAssure Verification", page_icon="🛡️", layout="centered")

st.markdown("""
    <style>
        .reportview-container {
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #0466c8;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }
        .stButton>button:hover {
            background-color: #0353a4;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 🛡️ Welcome to **IDAssure**")
st.markdown("#### ✅ Fast • Secure • Face + Document Verification")

with st.expander("ℹ️ What does this do?", expanded=False):
    st.markdown("""
    This app allows you to:
    - Upload your Aadhar (Image or PDF)
    - Capture a live selfie
    - Automatically detect DOB and estimate your age
    - Check face similarity using deep learning

    **Built using EasyOCR + FaceNet + Streamlit.**
    """)

# Upload and Capture
st.markdown("### 📤 Upload your Aadhar Document")
aadhar_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])

st.markdown("### 🤳 Capture a Selfie")
selfie_file = st.camera_input("")

if st.button("🔍 Verify Identity"):
    if not aadhar_file or not selfie_file:
        st.warning("⚠️ Please upload both Aadhar and Selfie to continue.")
    else:
        with st.spinner("🔎 Verifying... Please wait..."):
            if aadhar_file.type == "application/pdf":
                aadhar_img = extract_image_from_pdf(aadhar_file)
            else:
                aadhar_img = Image.open(aadhar_file).convert("RGB")

            selfie_img = Image.open(selfie_file).convert("RGB")

            emb1 = extract_face_embedding(aadhar_img)
            emb2 = extract_face_embedding(selfie_img)
            score = compare_faces(emb1, emb2)

            dob_text = extract_dob_text(aadhar_img)
            age = parse_age_from_dob(dob_text) if dob_text else None

            st.markdown("---")
            st.subheader("🧾 Verification Results")

            col1, col2 = st.columns(2)
            with col1:
                st.image(aadhar_img, caption="Aadhar Image", use_column_width=True)
            with col2:
                st.image(selfie_img, caption="Captured Selfie", use_column_width=True)

            st.write(f"👤 **Face Match:** `{score*100:.2f}%`")
            st.write(f"📅 **DOB Extracted:** `{dob_text if dob_text else 'Not Found'}`")
            st.write(f"🎂 **Estimated Age:** `{age if age else 'Not Found'}`")

            if score > 0.75 and age and age >= 18:
                st.success("✅ Identity & Age Verified")
            elif score > 0.75:
                st.warning("✅ Identity Verified, but Age < 18")
            elif score > 0.5:
                st.warning("⚠ Face match is low. Verification uncertain.")
            else:
                st.error("❌ Verification Failed")

