# EasyOCR-VehiclePlate

A robust OCR system for detecting and recognizing vehicle license plates. Built using computer vision and deep learning techniques.

## Features

- License plate detection using Haar Cascades
- Text extraction using EasyOCR (ResNet + CTC model)
- Image preprocessing for improved OCR accuracy
- Post-processing for text cleanup

## Requirements

- Python 3.8+
- OpenCV
- EasyOCR
- NumPy
- Matplotlib
- imutils

## Installation

1. Clone repository:
```bash
git clone https://github.com/TarunSamala/EasyOCR-VehiclePlate.git
```

## Limitations

- Works best with front-facing plate images
- Requires minimum 800px width for good recognition
- Performance varies with image quality and lighting
