"""
Text Classification Module
This module handles toxic content classification using RoBERTa
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class TextClassifier:
    def __init__(self):
        """Initialize text classification model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading text classification model on {self.device}...")
        
        # Using a more reliable model: s-nlp/roberta-base-toxicity-classifier
        # This model is specifically trained on toxic content with better accuracy
        model_name = "s-nlp/roberta-base-toxicity-classifier"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("RoBERTa toxicity classification model loaded successfully!")
    
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
                # Get probabilities using softmax
                probs = F.softmax(outputs.logits, dim=-1)
                predictions = probs.cpu().numpy()[0]
            
            # Model returns [neutral, toxic] probabilities
            neutral_score = float(predictions[0])
            toxic_score = float(predictions[1])
            
            # Determine classification with a threshold of 0.5
            if toxic_score > 0.5:
                classification = "toxic"
                confidence = toxic_score
            else:
                classification = "non-toxic"
                confidence = neutral_score
            
            # Create detailed scores
            detailed_scores = {
                'Toxic': toxic_score,
                'Severe Toxic': max(0.0, (toxic_score - 0.7) * 3),  # Only if very toxic
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
