# IDAssure – Smart Identity Verification App

**IDAssure** is an AI-powered identity verification web application built with **Streamlit**.  
It uses **face matching** and **OCR (Optical Character Recognition)** to verify identity from an Aadhar card (image or PDF) and a live selfie. It also extracts the date of birth and estimates age to support eligibility checks.

---

## 🔗 Live Demo

 Try it out here:  
**[IDAssure Web App](https://app-idassure-new-igdtuw.streamlit.app)**

---

##  Key Features

-  Upload **Aadhar card** (image or PDF format)  
-  Capture **live selfie** using webcam  
- **Face matching** using deep learning (pretrained model)  
-  Extract **Date of Birth** using OCR  
-  Estimate **user’s age** based on DOB  
-  Check **identity & age-based eligibility**

---

## Tech Stack

| Area               | Tools/Packages Used                              |
|--------------------|--------------------------------------------------|
| **Web Framework**   | Streamlit                                        |
| **Face Recognition**| facenet-pytorch, torch, torchvision             |
| **OCR Engine**      | easyocr                                          |
| **PDF Processing**  | PyMuPDF                                          |
| **Image Handling**  | OpenCV, Pillow                                   |
| **Others**          | numpy, scikit-image, python-bidi, PyYAML        |

---

##  Purpose

IDAssure was developed as a smart verification solution for Aadhar-based identity validation, combining computer vision and OCR techniques in a user-friendly interface.  
It can be useful for applications like **age-restricted platform onboarding**, **KYC automation**, or **eligibility checks** in digital services.

---



