# IDAssure: From Image to Face, We Trace with Grace

**IDAssure** is a smart identity verification system developed by **Team Pretty Pixels** at **Indira Gandhi Delhi Technical University for Women (IGDTUW)**. The platform verifies user identity and age using a combination of Aadhar document analysis and live facial recognition.

Deployed here: [https://ida-ssure-by-team-pretty-pixels.vercel.app](https://ida-ssure-by-team-pretty-pixels.vercel.app)

YouTube Tutorial: [https://youtu.be/0k3FhwTyRK4](https://youtu.be/0k3FhwTyRK4)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Screenshots](#screenshots)
- [Installation Guide](#installation-guide)
- [Future Scope](#future-scope)
- [About the Team](#about-the-team)
- [License](#license)

---

## Project Overview

IDAssure simplifies identity verification by allowing users to upload their Aadhar card (image or PDF) and capture a selfie. The system then performs facial recognition and OCR-based DOB extraction to confirm both identity and age. It is suitable for portals requiring KYC, digital onboarding, or age-restricted services.

---

## Live Demo

- Web App: [https://ida-ssure-by-team-pretty-pixels.vercel.app](https://ida-ssure-by-team-pretty-pixels.vercel.app)

---

## Features

- Face matching using MTCNN and FaceNet
- OCR-based DOB extraction from Aadhar using EasyOCR
- Age estimation based on DOB
- PDF and image upload support
- PAN card support added
- Modern React dashboard with sidebar navigation
- Streamlit-based AI backend with JSON API

---

## How It Works

1. **User Input**  
   - Upload Aadhar Card or PAN Card (Image or PDF)  
   - Capture live selfie using webcam  

2. **Face Detection and Comparison**  
   - Detects face in both inputs using MTCNN  
   - Extracts 512-d embeddings using InceptionResNetV1  
   - Compares faces using cosine similarity  

3. **DOB Extraction and Age Estimation**  
   - Uses EasyOCR to read text from document  
   - Applies regex to extract date formats  
   - Computes estimated age using datetime  

4. **Verification Logic**  
   - If Match > 75% and Age ≥ 18 → Identity Verified  
   - If Match > 75% but Age < 18 → Underage Warning  
   - If Match < 50% → Identity Rejected  

5. **Output**  
   - Face Match Percentage  
   - Extracted DOB  
   - Estimated Age  
   - Final Verdict (Verified / Underage / Rejected)  

---

## Tech Stack

### Backend (ML & OCR)
- Python  
- Streamlit  
- Facenet-Pytorch (MTCNN + InceptionResNetV1)  
- EasyOCR  
- PyMuPDF  
- NumPy, Pillow, OpenCV  
- Firebase  

### Frontend (UI)
- React.js  
- TailwindCSS  
- Webcam.js  
- React Router  
- Fetch API  

---

## Screenshots

| Upload Aadhar & Selfie | Result Page | Dashboard |
|------------------------|-------------|-----------|
| ![Upload](./screenshots/upload.png) | ![Result](./screenshots/result.png) | ![Dashboard](./screenshots/dashboard.png) |

---

## Installation Guide

If you download the IDAssure ZIP file to your system, follow these steps:

1. Open terminal and navigate to the backend folder where `app.py` exists  
   Example:  
   `cd C:\Users\shant\Downloads\client\IDAssure\backend`

2. Run the backend:  
   `python app.py`

   You should see something like:  
   `PS C:\Users\shant\Downloads\client\IDAssure\backend> python app.py`

3. Open a new terminal tab and navigate to the frontend folder  
   Example:  
   `cd ..\frontend`

4. Start the frontend using:  
   `npm run dev`

5. Open the link provided in the terminal  
   Example:  
   `http://localhost:5173/`

6. The application will open in your browser.

7. Refer to the YouTube video for additional help.

---

## Future Scope

- Multilingual OCR  
- Smart document classification  
- Third-party API integration for SaaS use cases  
- Blockchain-based identity ledger  

---

## About the Team

This project was developed by **Team Pretty Pixels** from **IGDTUW**.

Team Members:
- Shanti Kumari  
- Sneha Kumari  
- Sadhya Gupta

---

## License

This project is licensed under the **Apache License 2.0**.
