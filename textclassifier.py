"""
Text Classification Module
Simple keyword-based toxic content classifier
"""

import re

class TextClassifier:
    def __init__(self):
        """Initialize keyword-based classifier"""
        print("Loading keyword-based toxic content classifier...")
        
        # Comprehensive toxic keywords list
        self.toxic_keywords = {
            # Hate & aggression
            'hate': 0.8, 'hating': 0.8, 'hated': 0.8,
            'kill': 0.9, 'die': 0.7, 'death': 0.6,
            'destroy': 0.6, 'hurt': 0.6,
            
            # Insults
            'stupid': 0.7, 'idiot': 0.8, 'moron': 0.8,
            'dumb': 0.7, 'fool': 0.6, 'foolish': 0.6,
            'ignorant': 0.5, 'worthless': 0.8,
            'useless': 0.7, 'pathetic': 0.7,
            'loser': 0.7, 'trash': 0.7, 'garbage': 0.7,
            
            # Profanity
            'damn': 0.6, 'hell': 0.5, 'shit': 0.8,
            'fuck': 0.9, 'fucking': 0.9, 'bitch': 0.9,
            'bastard': 0.8, 'ass': 0.7, 'asshole': 0.9,
            
            # Appearance/personal attacks
            'ugly': 0.6, 'fat': 0.5, 'disgusting': 0.7,
            
            # Threats
            'threat': 0.7, 'threaten': 0.7,
            'attack': 0.6, 'violence': 0.7
        }
        
        # Toxic patterns (combinations that are clearly toxic)
        self.toxic_patterns = [
            (r'\bhate\s+you\b', 0.9),
            (r'\byou\s+are\s+(stupid|idiot|dumb|moron)', 0.9),
            (r'\bshut\s+up\b', 0.7),
            (r'\bgo\s+to\s+hell\b', 0.8),
            (r'\bfuck\s+you\b', 1.0),
            (r'\byou\s+suck\b', 0.7),
        ]
        
        print("Keyword-based classifier loaded successfully!")
    
    def classify_text(self, text):
        """
        Classify text for toxic content using keywords
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Classification results with probabilities
        """
        try:
            text_lower = text.lower()
            max_score = 0.0
            found_keywords = []
            
            # Check patterns first (highest priority)
            for pattern, score in self.toxic_patterns:
                if re.search(pattern, text_lower):
                    max_score = max(max_score, score)
                    found_keywords.append(f"pattern:{pattern}")
            
            # Check individual keywords
            for keyword, score in self.toxic_keywords.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    max_score = max(max_score, score)
                    found_keywords.append(keyword)
            
            # Calculate final toxic score
            if max_score > 0:
                # If multiple keywords found, increase score slightly
                keyword_count = len(found_keywords)
                bonus = min((keyword_count - 1) * 0.1, 0.2)
                toxic_score = min(max_score + bonus, 1.0)
            else:
                toxic_score = 0.0
            
            non_toxic_score = 1.0 - toxic_score
            
            # Classification with threshold
            if toxic_score >= 0.5:
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
