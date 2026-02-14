"""
Image Captioning Module using BLIP (Optimized for Streamlit Cloud)
This module handles image captioning functionality for the toxic content classification project.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageCaptioner:
    def __init__(self):
        """Initialize BLIP model and processor (using lighter BLIP-1)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BLIP model on {self.device}...")
        
        # Using BLIP-1 base model (lighter than BLIP-2, better for deployment)
        model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate_caption(self, image):
        """
        Generate caption for an image
        
        Args:
            image: PIL Image object or path to image
            
        Returns:
            str: Generated caption
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image)
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Process image and generate caption
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
            
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            return caption
            
        except Exception as e:
            return f"Error generating caption: {str(e)}"

# Singleton instance
_captioner_instance = None

def get_captioner():
    """Get or create ImageCaptioner instance (singleton pattern)"""
    global _captioner_instance
    if _captioner_instance is None:
        _captioner_instance = ImageCaptioner()
    return _captioner_instance
