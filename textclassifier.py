"""
Text Classification Module
This module handles toxic content classification
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class TextClassifier:
    def __init__(self):
        """Initialize text classification model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading text classification model on {self.device}...")
        
        # Using cardiffnlp offensive language classifier - very accurate
        model_name = "cardiffnlp/twitter-roberta-base-offensive"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Model labels: not-offensive, offensive
        self.labels = ['not-offensive', 'offensive']
        
        print("RoBERTa offensive language classifier loaded successfully!")
    
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
            
            # Model returns [not-offensive, offensive] probabilities
            not_offensive_score = float(predictions[0])
            offensive_score = float(predictions[1])
            
            # Determine classification
            # Using lower threshold (0.3) to catch more toxic content
            if offensive_score > 0.3:
                classification = "toxic"
                confidence = offensive_score
            else:
                classification = "non-toxic"
                confidence = not_offensive_score
            
            # Create detailed scores
            detailed_scores = {
                'Toxic': offensive_score,
                'Severe Toxic': max(0.0, (offensive_score - 0.7) * 2.5),
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
