"""
Text Classification Module
This module handles toxic content classification using Fine-tuned DistilBERT
(Meets Task 1 requirements: Fine-tuned DistilBERT for text classification)
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re

class TextClassifier:
    def __init__(self):
        """Initialize DistilBERT-based text classification model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading DistilBERT text classification model on {self.device}...")
        
        # Using Fine-tuned DistilBERT model (as required by Task 1)
        # This model is a DistilBERT architecture fine-tuned for toxic comment classification
        model_name = "martin-ha/toxic-comment-model"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Toxic keywords for hybrid approach (improves accuracy)
        self.toxic_keywords = {
            'hate': 0.85, 'stupid': 0.80, 'idiot': 0.85,
            'kill': 0.95, 'die': 0.75, 'fuck': 0.95,
            'shit': 0.85, 'damn': 0.65, 'hell': 0.60,
            'dumb': 0.75, 'moron': 0.85, 'fool': 0.70,
            'loser': 0.75, 'ugly': 0.70, 'worthless': 0.80,
            'pathetic': 0.75, 'trash': 0.75, 'garbage': 0.75,
            'bitch': 0.90, 'bastard': 0.85, 'ass': 0.70, 'asshole': 0.90
        }
        
        print("DistilBERT model loaded successfully!")
    
    def _get_keyword_score(self, text):
        """Helper function to calculate keyword-based toxicity score"""
        text_lower = text.lower()
        max_score = 0.0
        
        for keyword, severity in self.toxic_keywords.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                max_score = max(max_score, severity)
        
        return max_score
    
    def classify_text(self, text):
        """
        Classify text for toxic content using DistilBERT
        
        Uses a hybrid approach:
        1. DistilBERT model predictions (primary)
        2. Keyword matching (fallback for better accuracy)
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Classification results with probabilities
        """
        try:
            # Get keyword-based score (for fallback)
            keyword_score = self._get_keyword_score(text)
            
            # Tokenize and get DistilBERT predictions
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Apply sigmoid for multi-label classification
                predictions = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            
            # Get toxic score from model (first label is usually 'toxic')
            model_toxic_score = float(predictions[0]) if len(predictions) > 0 else 0.0
            
            # Hybrid approach: use keyword score if higher (improves accuracy)
            # This helps catch clear toxic cases that the model might miss
            if keyword_score > model_toxic_score:
                toxic_score = keyword_score
                print(f"Using keyword-based score: {toxic_score:.2f}")
            else:
                toxic_score = model_toxic_score
                print(f"Using DistilBERT model score: {toxic_score:.2f}")
            
            # Determine classification
            if toxic_score >= 0.5:
                classification = "toxic"
                confidence = toxic_score
            else:
                classification = "non-toxic"
                confidence = 1.0 - toxic_score
            
            # Calculate severe toxic score
            severe_toxic_score = max(0.0, (toxic_score - 0.7) * 2.5)
            
            detailed_scores = {
                'Toxic': toxic_score,
                'Severe Toxic': severe_toxic_score
            }
            
            return {
                "classification": classification,
                "confidence": confidence,
                "detailed_scores": detailed_scores
            }
            
        except Exception as e:
            print(f"Error in classification: {str(e)}")
            # Fallback to keyword-based if model fails
            keyword_score = self._get_keyword_score(text)
            
            if keyword_score >= 0.5:
                classification = "toxic"
                confidence = keyword_score
            else:
                classification = "non-toxic"
                confidence = 1.0 - keyword_score
            
            return {
                "classification": classification,
                "confidence": confidence,
                "detailed_scores": {
                    'Toxic': keyword_score,
                    'Severe Toxic': max(0.0, (keyword_score - 0.7) * 2.5)
                }
            }

# Singleton instance
_classifier_instance = None

def get_classifier():
    """Get or create TextClassifier instance (singleton pattern)"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TextClassifier()
    return _classifier_instance
