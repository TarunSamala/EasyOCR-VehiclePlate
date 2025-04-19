# EasyOCR-VehiclePlate 🚗🔍

A robust OCR system for detecting and recognizing vehicle license plates with special support for Indian formats. Built using computer vision and deep learning techniques.

## Features ✨

- **Automatic Dataset Processing**
  - XML annotation parsing
  - License plate extraction with bounding boxes
- **Advanced OCR Pipeline**
  - Haar Cascade/YOLO plate detection
  - EasyOCR text extraction (ResNet + CTC model)
  - Indian plate format validation (XX00XX0000)
- **Preprocessing**
  - CLAHE contrast enhancement
  - Non-local means denoising
  - Adaptive thresholding
- **Output Generation**
  - Extracted plate images
  - Formatted text files with country tagging
  - Annotated images with bounding boxes

## Requirements 📋

- Python 3.8+
- OpenCV
- EasyOCR
- NumPy
- Matplotlib
- imutils
- xml.etree.ElementTree (built-in)
- re (built-in)

## Installation ⚙️

```bash
git clone https://github.com/TarunSamala/EasyOCR-VehiclePlate.git
pip install -r requirements.txt