import os
import cv2
import easyocr
from pathlib import Path

class PlateTextExtractor:
    def __init__(self, gpu=False):
        """
        Initialize text extractor for license plates
        :param gpu: Boolean flag for GPU acceleration
        """
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.output_file = "plate_texts.txt"
        self.confidence_threshold = 0.4  # Minimum confidence to consider
        
    def preprocess_image(self, image):
        """Enhance plate image for better OCR results"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, thresh = cv2.threshold(denoised, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh
    
    def process_plate(self, image_path):
        """
        Process a single license plate image
        :return: (text, confidence)
        """
        image = cv2.imread(image_path)
        if image is None:
            return ("Invalid image", 0.0)
            
        processed = self.preprocess_image(image)
        results = self.reader.readtext(processed, detail=1)
        
        if not results:
            return ("No text detected", 0.0)
        
        # Filter and sort results by horizontal position
        valid_results = [r for r in results if r[2] >= self.confidence_threshold]
        sorted_results = sorted(valid_results, 
                              key=lambda x: x[0][0][0])  # Sort by left-most x
        
        # Combine text and calculate average confidence
        texts = []
        confidences = []
        for res in sorted_results:
            texts.append(res[1])
            confidences.append(res[2])
        
        combined_text = self.clean_text(texts)
        avg_confidence = sum(confidences)/len(confidences) if confidences else 0.0
        return (combined_text, avg_confidence)
    
    def clean_text(self, text_list):
        """Clean and format detected text"""
        full_text = ''.join(text_list).upper()
        # Remove special characters while preserving alphanumerics
        return ''.join(c for c in full_text if c.isalnum())
    
    def process_directory(self, plates_dir):
        """Process all images in directory"""
        results = []
        
        for filename in sorted(os.listdir(plates_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(plates_dir, filename)
                text, confidence = self.process_plate(filepath)
                results.append({
                    'filename': filename,
                    'text': text,
                    'confidence': round(confidence, 3)
                })
        
        # Save results
        with open(self.output_file, 'w') as f:
            f.write("Filename|Detected Text|Confidence\n")
            f.write("-"*40 + "\n")
            for res in results:
                line = f"{res['filename']}|{res['text']}|{res['confidence']}\n"
                f.write(line)
        
        print(f"Processed {len(results)} plates. Results saved to {self.output_file}")

if __name__ == "__main__":
    # Initialize with GPU if available
    extractor = PlateTextExtractor(gpu=True)
    
    # Process extracted plates directory
    extractor.process_directory("extracted_plates")