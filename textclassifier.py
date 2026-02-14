"""
Text Classification Module
This module handles toxic content classification using Zero-Shot Classification
"""

from transformers import pipeline
import re

class TextClassifier:
    def __init__(self):
        """Initialize zero-shot classification model"""
        print(f"Loading zero-shot classification model...")
        
        # Using zero-shot classification - most flexible and accurate
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        
        # Toxic keywords for additional checking
        self.toxic_keywords = [
            'hate', 'stupid', 'idiot', 'dumb', 'fool', 'moron',
            'kill', 'die', 'death', 'damn', 'hell', 'shit',
            'fuck', 'bitch', 'ass', 'bastard', 'loser', 'ugly',
            'worthless', 'useless', 'pathetic', 'trash', 'garbage'
        ]
        
        print("Zero-shot classification model loaded successfully!")
    
    def classify_text(self, text):
        """
        Classify text for toxic content
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Classification results with probabilities
        """
        try:
            # Convert to lowercase for checking
            text_lower = text.lower()
            
            # Check for toxic keywords
            keyword_score = 0
            found_keywords = []
            for keyword in self.toxic_keywords:
                if keyword in text_lower:
                    keyword_score += 1
                    found_keywords.append(keyword)
            
            # Normalize keyword score (0-1)
            keyword_toxicity = min(keyword_score * 0.3, 1.0)
            
            # Use zero-shot classification
            candidate_labels = ["friendly positive message", "toxic offensive hateful message"]
            result = self.classifier(text, candidate_labels)
            
            # Get toxic probability (index 1 = toxic)
            if result['labels'][0] == "toxic offensive hateful message":
                model_toxic_score = result['scores'][0]
            else:
                model_toxic_score = result['scores'][1]
            
            # Combine both scores (weighted average)
            # Give more weight to keyword matching for clear cases
            if keyword_score > 0:
                toxic_score = (keyword_toxicity * 0.6) + (model_toxic_score * 0.4)
            else:
                toxic_score = model_toxic_score
            
            # Ensure score is between 0 and 1
            toxic_score = min(max(toxic_score, 0.0), 1.0)
            non_toxic_score = 1.0 - toxic_score
            
            # Classification with threshold
            if toxic_score > 0.4:  # Lower threshold for better detection
                classification = "toxic"
                confidence = toxic_score
            else:
                classification = "non-toxic"
                confidence = non_toxic_score
            
            # Create detailed scores
            detailed_scores = {
                'Toxic': toxic_score,
                'Severe Toxic': max(0.0, (toxic_score - 0.7) * 2.0),
            }
            
            # Add debug info if keywords found
            debug_info = ""
            if found_keywords:
                debug_info = f" (Found keywords: {', '.join(found_keywords)})"
            
            return {
                "classification": classification,
                "confidence": confidence,
                "detailed_scores": detailed_scores,
                "debug": debug_info
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
