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
        
        # Using DistilBERT fine-tuned for toxic comment classification
        # This model is based on DistilBERT architecture (meets Task 1 requirements)
        model_name = "martin-ha/toxic-comment-model"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Classification labels
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        print("DistilBERT text classification model loaded successfully!")
    
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
                predictions = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            
            # Create results dictionary
            results = {}
            for label, score in zip(self.labels, predictions):
                results[label] = float(score)
            
            # Determine overall classification
            max_score = max(results.values())
            if max_score > 0.5:
                classification = max(results, key=results.get)
                confidence = max_score
            else:
                classification = "non-toxic"
                confidence = 1 - max_score
            
            return {
                "classification": classification,
                "confidence": confidence,
                "detailed_scores": results
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
