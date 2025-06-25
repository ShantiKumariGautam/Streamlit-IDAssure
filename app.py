import streamlit as st
from PIL import Image
from datetime import datetime
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import easyocr
import re
import fitz  # PyMuPDF
import torch

# --- Model Loading (Cached) ---
# Use st.cache_resource to load models only once and keep them in memory.

@st.cache_resource
def load_mtcnn_model():
    """Loads the MTCNN face detection model."""
    # Check for CUDA availability, but default to CPU for deployment
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return MTCNN(image_size=160, margin=20, device=device)

@st.cache_resource
def load_resnet_model():
    """Loads the InceptionResnetV1 face recognition model."""
    # The model will be loaded to the device specified during MTCNN loading if passed
    return InceptionResnetV1(pretrained='vggface2').eval()

@st.cache_resource
def load_ocr_reader():
    """Loads the EasyOCR reader model."""
    # Using gpu=False is crucial for CPU-only environments like Streamlit Cloud
    return easyocr.Reader(['en'], gpu=False)

# Load all models at the start
mtcnn = load_mtcnn_model()
resnet = load_resnet_model()
reader = load_ocr_reader()

# --- Core Functions ---

def extract_face_embedding(image: Image.Image):
    """Detects a face and extracts its embedding using the loaded models."""
    face = mtcnn(image)
    if face is not None:
        # Move tensor to CPU for numpy conversion
        return resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten()
    return None

def compare_faces(emb1, emb2):
    """Compares two face embeddings using cosine similarity."""
    if emb1 is None or emb2 is None:
        return 0.0
    # Normalize embeddings to unit vectors
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    # Return the dot product (cosine similarity)
    return float(np.dot(emb1, emb2))

def extract_image_from_pdf(pdf_file):
    """Extracts the first image found in an uploaded PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            pix = page.get_pixmap()
            # Check if the pixmap has samples (is not empty)
            if pix.samples:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return img
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return None

def extract_dob_text(image: Image.Image):
    """Extracts date of birth text from an image using OCR."""
    results = reader.readtext(np.array(image))
    text = " ".join([item[1] for item in results])
    
    # More robust regex patterns to find DOB
    patterns = [
        r"DOB\s*[:\s-]\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"Date of Birth\s*[:\s-]\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
        r"\b(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4})\b"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Return the first captured group if it exists, otherwise the full match
            return match.group(1) if match.groups() else match.group(0)
    
    # As a fallback, find any 4-digit number that could be a birth year
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if year_match:
        return year_match.group(0)
        
    return None

def parse_age_from_dob(dob_text):
    """Parses a DOB string and calculates the age."""
    if not dob_text:
        return None
        
    # Standard date formats
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", 
        "%d %B %Y", "%d %b %Y", "%d %m %Y"
    ]
    
    # Handle year-only case
    if dob_text.isdigit() and len(dob_text) == 4:
        try:
            dob_year = int(dob_text)
            today = datetime.today()
            return today.year - dob_year
        except ValueError:
            return None

    for fmt in formats:
        try:
            dob = datetime.strptime(dob_text.strip(), fmt)
            today = datetime.today()
            # Calculate age accurately
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            continue
            
    return None

# --- Streamlit UI ---

st.set_page_config(page_title="IDAssure", page_icon="🛡️", layout="wide")
st.title("🛡️ IDAssure: Smart Identity Verification Portal")

col1, col2 = st.columns(2)

with col1:
    st.header("Step 1: Upload ID Document")
    aadhar_file = st.file_uploader("Upload Aadhar or ID Card (Image or PDF)", type=["jpg", "jpeg", "png", "pdf"])

with col2:
    st.header("Step 2: Live Selfie Capture")
    selfie_file = st.camera_input("Take your selfie to match against the ID")

if st.button("✅ Verify Identity", type="primary", use_container_width=True):
    if not aadhar_file or not selfie_file:
        st.warning("Please upload an ID document and take a selfie to proceed.")
    else:
        with st.spinner("Analyzing documents... This may take a moment."):
            
            # --- Image Processing ---
            if aadhar_file.type == "application/pdf":
                aadhar_img = extract_image_from_pdf(aadhar_file)
            else:
                aadhar_img = Image.open(aadhar_file).convert("RGB")
            
            selfie_img = Image.open(selfie_file).convert("RGB")

            if aadhar_img is None:
                st.error("Could not extract an image from the provided PDF. Please try another file.")
            else:
                # --- Analysis ---
                emb1 = extract_face_embedding(aadhar_img)
                emb2 = extract_face_embedding(selfie_img)
                score = compare_faces(emb1, emb2)
                
                dob_text = extract_dob_text(aadhar_img)
                age = parse_age_from_dob(dob_text)

                # --- Display Results ---
                st.subheader("🔍 Verification Results")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.image(aadhar_img, caption="ID Document Image", use_column_width=True)
                with res_col2:
                    st.image(selfie_img, caption="Live Selfie", use_column_width=True)

                face_match_threshold = 0.70  # Adjusted for better real-world matching
                
                # Face Match Result
                st.metric(label="👤 Face Match Confidence", value=f"{score*100:.2f}%")
                if emb1 is None:
                     st.error("Could not detect a face on the ID document.")
                elif emb2 is None:
                     st.error("Could not detect a face in the selfie.")
                elif score > face_match_threshold:
                    st.success("Face match successful.")
                else:
                    st.error("Faces do not match.")
                
                # Age Verification Result
                st.metric(label="🎂 Estimated Age", value=str(age) if age else "Not Found")
                if dob_text:
                    st.info(f"Detected Date of Birth text: **{dob_text}**")
                
                if age is not None and age >= 18:
                    st.success("Age verification successful (18+).")
                elif age is not None:
                    st.warning("Age verification failed (under 18).")
                else:
                    st.error("Could not determine age from the document.")

                # Final Verdict
                st.subheader("Final Verdict")
                if score > face_match_threshold and age is not None and age >= 18:
                    st.success("✅ Identity and Age Verified Successfully!")
                else:
                    st.error("❌ Verification Failed. Please review the results above.")
