import os
import cv2
import easyocr
import re
from pathlib import Path

class IndianPlateExtractor:
    def __init__(self, gpu=False):
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.output_file = "indian_plates.txt"
        self.plate_pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')
        
    def preprocess_image(self, image):
        """Optimized preprocessing for Indian plates"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        _, thresh = cv2.threshold(enhanced, 0, 255, 
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh
    
    def validate_indian_format(self, text):
        """Validate against Indian license plate pattern"""
        return bool(self.plate_pattern.match(text))
    
    def clean_text(self, ocr_results):
        """Indian plate specific cleaning"""
        raw_text = ''.join(ocr_results).upper()
        # Remove special characters and enforce Indian format
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text)
        # Format with spaces for better readability: MH 02 DE 1433
        formatted = re.sub(r'^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$', 
                         r'\1 \2 \3 \4', cleaned)
        return formatted.replace(' ', '')  # Return without spaces
    
    def process_plate(self, image_path):
        """Process single plate with Indian format validation"""
        image = cv2.imread(image_path)
        if image is None:
            return "Invalid Image"
            
        processed = self.preprocess_image(image)
        results = self.reader.readtext(processed, detail=0)
        
        if not results:
            return "No Text Detected"
        
        cleaned_text = self.clean_text(results)
        if self.validate_indian_format(cleaned_text):
            return cleaned_text
        return "Invalid Indian Format"
    
    def process_directory(self, plates_dir):
        """Process all plates and save results with country tag"""
        Path(plates_dir).mkdir(exist_ok=True)
        
        with open(self.output_file, 'w') as f:
            f.write("Filename|Plate Number|Country\n")
            f.write("-"*40 + "\n")
            
            for filename in sorted(os.listdir(plates_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(plates_dir, filename)
                    plate_text = self.process_plate(filepath)
                    
                    if plate_text not in ["Invalid Image", "No Text Detected", "Invalid Indian Format"]:
                        line = f"{filename}|{plate_text}|Indian\n"
                        f.write(line)

if __name__ == "__main__":
    extractor = IndianPlateExtractor(gpu=False)
    extractor.process_directory("extracted_plates")
    print(f"Indian plate results saved to {extractor.output_file}")