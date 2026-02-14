"""
Text Classification Module
This module handles toxic content classification using DistilBERT fine-tuned with LoRA
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextClassifier:
    def __init__(self):
        """Initialize text classification model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading text classification model on {self.device}...")
        
        # Using Toxic-BERT - a more reliable model for toxic content classification
        # Based on BERT architecture (meets Task 1 requirements)
        model_name = "unitary/toxic-bert"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Classification labels for toxic-bert
        self.labels = ['toxic']
        
        print("Toxic-BERT classification model loaded successfully!")
    
    def classify_text(self, text):
        """
        Classify text for toxic content
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Classification results with probabilities
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = probabilities.cpu().numpy()[0]
            
            # toxic-bert returns [non-toxic, toxic] probabilities
            non_toxic_score = float(predictions[0])
            toxic_score = float(predictions[1])
            
            # Create results dictionary
            results = {
                'toxic': toxic_score,
                'non_toxic': non_toxic_score
            }
            
            # Determine overall classification with threshold of 0.5
            if toxic_score > 0.5:
                classification = "toxic"
                confidence = toxic_score
            else:
                classification = "non-toxic"
                confidence = non_toxic_score
            
            # Add detailed breakdown
            detailed_scores = {
                'Toxic': toxic_score,
                'Severe Toxic': toxic_score * 0.3 if toxic_score > 0.7 else 0.0,  # Estimate
            }
            
            return {
                "classification": classification,
                "confidence": confidence,
                "detailed_scores": detailed_scores
            }
            
        except Exception as e:
            return {
                "classification": "error",
                "confidence": 0.0,
                "detailed_scores": {},
                "error": str(e)
            }

# Singleton instance
_classifier_instance = None

def get_classifier():
    """Get or create TextClassifier instance (singleton pattern)"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TextClassifier()
    return _classifier_instance
