"""
Text Classification Module
Simple keyword-based toxic content classifier
"""

import re

class TextClassifier:
    def __init__(self):
        """Initialize keyword-based classifier"""
        print("Loading keyword-based toxic content classifier...")
        
        # Toxic keywords with their severity scores
        self.toxic_keywords = {
            'hate': 0.85,
            'stupid': 0.80,
            'idiot': 0.85,
            'kill': 0.95,
            'die': 0.75,
            'fuck': 0.95,
            'shit': 0.85,
            'damn': 0.65,
            'hell': 0.60,
            'dumb': 0.75,
            'moron': 0.85,
            'fool': 0.70,
            'loser': 0.75,
            'ugly': 0.70,
            'worthless': 0.80,
            'pathetic': 0.75,
            'trash': 0.75,
            'garbage': 0.75,
            'bitch': 0.90,
            'bastard': 0.85,
            'ass': 0.70,
            'asshole': 0.90
        }
        
        print("Classifier ready!")
    
    def classify_text(self, text):
        """Classify text for toxic content"""
        if not text or not text.strip():
            return {
                "classification": "non-toxic",
                "confidence": 1.0,
                "detailed_scores": {"Toxic": 0.0, "Severe Toxic": 0.0}
            }
        
        text_lower = text.lower()
        toxic_score = 0.0
        found_words = []
        
        # Check each toxic keyword
        for keyword, severity in self.toxic_keywords.items():
            # Use word boundary for exact word matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                toxic_score = max(toxic_score, severity)
                found_words.append(keyword)
        
        # Boost score if multiple toxic words found
        if len(found_words) > 1:
            toxic_score = min(toxic_score + 0.1, 1.0)
        
        # Determine classification
        if toxic_score >= 0.5:
            classification = "toxic"
            confidence = toxic_score
        else:
            classification = "non-toxic"
            confidence = 1.0 - toxic_score
        
        # Calculate severe toxic score
        severe_toxic_score = max(0.0, (toxic_score - 0.8) * 5.0)
        
        result = {
            "classification": classification,
            "confidence": confidence,
            "detailed_scores": {
                "Toxic": toxic_score,
                "Severe Toxic": severe_toxic_score
            }
        }
        
        # Debug: print what was found
        if found_words:
            print(f"DEBUG: Found toxic words: {found_words}, Score: {toxic_score}")
        
        return result

# Singleton
_classifier_instance = None

def get_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TextClassifier()
    return _classifier_instance
