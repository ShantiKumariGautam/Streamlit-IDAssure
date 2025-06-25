# IDAssure – Smart Identity Verification App

**IDAssure** is a Streamlit-based web application that verifies identity using face matching and OCR from Aadhar (image or PDF) and a live selfie. It checks face similarity and extracts date of birth to estimate age.

---

## Live App

Access the app here:  
**[https://app-idassure-new-igdtuw.streamlit.app](https://app-idassure-new-igdtuw.streamlit.app)**

---

## Features

- Upload Aadhar card (image or PDF)
- Take live selfie via webcam
- Face matching using pretrained deep learning model
- OCR-based DOB extraction
- Age estimation and eligibility check

---

## Technologies Used

- **Framework**: Streamlit
- **Face Recognition**: facenet-pytorch, torch, torchvision
- **OCR Engine**: easyocr
- **PDF Image Extraction**: PyMuPDF
- **Image Processing**: OpenCV, Pillow
- **Others**: numpy, scikit-image, python-bidi, PyYAML

---

