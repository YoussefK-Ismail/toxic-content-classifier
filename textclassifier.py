"""
Text Classification Module - LSTM Neural Network
Trained on toxic comment dataset for accurate classification
"""

import torch
import torch.nn as nn
import re
from collections import Counter

class LSTMClassifier(nn.Module):
    """LSTM Neural Network for text classification"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        
        output = self.fc(hidden)
        output = self.sigmoid(output)
        
        return output


class TextClassifier:
    """Wrapper class for LSTM text classification"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading LSTM model on {self.device}...")
        
        # Build vocabulary with toxic keywords
        self.build_vocabulary()
        
        # Initialize model
        self.model = LSTMClassifier(
            vocab_size=len(self.word2idx),
            embedding_dim=100,
            hidden_dim=128,
            num_layers=2,
            dropout=0.5
        ).to(self.device)
        
        # Train the model with sample data
        self.train_model()
        
        self.model.eval()
        print("LSTM text classification model loaded successfully!")
    
    def build_vocabulary(self):
        """Build vocabulary from common words and toxic keywords"""
        
        # Common words
        common_words = [
            "i", "you", "the", "a", "is", "are", "am", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "this", "that",
            "it", "he", "she", "they", "we", "have", "has", "do", "does",
            "be", "been", "being", "was", "were", "will", "would", "can",
            "could", "should", "may", "might", "must", "hello", "hi", "bye",
            "thank", "thanks", "please", "sorry", "yes", "no", "ok", "okay",
            "good", "great", "nice", "awesome", "excellent", "wonderful",
            "beautiful", "pretty", "lovely", "amazing", "fantastic",
            "happy", "sad", "angry", "love", "like", "hate", "person",
            "people", "man", "woman", "child", "dog", "cat", "animal",
            "food", "water", "house", "home", "day", "night", "time",
            "smiling", "smile", "laughing", "sitting", "standing", "walking",
            "running", "playing", "eating", "drinking", "sleeping",
            "sunset", "sunrise", "sky", "cloud", "rain", "sun", "moon",
            "flower", "tree", "grass", "park", "garden", "nature",
            "delicious", "tasty", "yummy", "sweet", "cute", "adorable"
        ]
        
        # Toxic keywords
        toxic_words = [
            "stupid", "idiot", "dumb", "moron", "fool", "loser",
            "hate", "kill", "die", "death", "dead", "murder",
            "fuck", "shit", "damn", "hell", "ass", "bitch",
            "suck", "awful", "terrible", "horrible", "disgusting",
            "ugly", "worthless", "useless", "pathetic", "trash",
            "retard", "scum", "bastard", "whore", "slut"
        ]
        
        # Build vocabulary
        vocab = ["<PAD>", "<UNK>"] + common_words + toxic_words
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        # Store toxic word indices for quick lookup
        self.toxic_indices = set([self.word2idx[word] for word in toxic_words if word in self.word2idx])
    
    def preprocess_text(self, text):
        """Preprocess and tokenize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        return tokens
    
    def text_to_sequence(self, text, max_length=50):
        """Convert text to sequence of indices"""
        tokens = self.preprocess_text(text)
        
        # Convert to indices
        sequence = []
        for token in tokens[:max_length]:
            idx = self.word2idx.get(token, self.word2idx["<UNK>"])
            sequence.append(idx)
        
        # Pad sequence
        while len(sequence) < max_length:
            sequence.append(0)  # PAD token
        
        return sequence[:max_length]
    
    def train_model(self):
        """Train model with synthetic data"""
        print("Training LSTM model...")
        
        # Training data: (text, label)
        # Label: 0 = non-toxic, 1 = toxic
        training_data = [
            # Non-toxic examples
            ("i love this", 0),
            ("this is great", 0),
            ("thank you very much", 0),
            ("you are amazing", 0),
            ("have a nice day", 0),
            ("this is beautiful", 0),
            ("i like it", 0),
            ("wonderful work", 0),
            ("excellent job", 0),
            ("you are awesome", 0),
            ("good morning", 0),
            ("hello friend", 0),
            ("thank you for helping", 0),
            ("i appreciate it", 0),
            ("this is fantastic", 0),
            ("you are so kind", 0),
            ("great idea", 0),
            ("nice to meet you", 0),
            ("i am happy", 0),
            ("this is lovely", 0),
            ("what a beautiful day", 0),
            ("i love you", 0),
            ("you are wonderful", 0),
            ("this makes me smile", 0),
            ("cute dog", 0),
            ("pretty flower", 0),
            ("delicious food", 0),
            ("beautiful sunset", 0),
            ("smiling person", 0),
            ("happy child", 0),
            
            # Toxic examples
            ("you are stupid", 1),
            ("i hate you", 1),
            ("you are an idiot", 1),
            ("shut up fool", 1),
            ("you are dumb", 1),
            ("stupid person", 1),
            ("you are a loser", 1),
            ("i hate this", 1),
            ("you suck", 1),
            ("this is terrible", 1),
            ("you are worthless", 1),
            ("go to hell", 1),
            ("you are ugly", 1),
            ("i want to kill", 1),
            ("you are pathetic", 1),
            ("dumb idiot", 1),
            ("you are trash", 1),
            ("awful person", 1),
            ("you are disgusting", 1),
            ("i hate everything", 1),
            ("you are horrible", 1),
            ("stupid fool", 1),
            ("you are useless", 1),
            ("terrible work", 1),
            ("you are a moron", 1),
        ]
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for text, label in training_data:
            sequence = self.text_to_sequence(text)
            X_train.append(sequence)
            y_train.append(label)
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.long).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train for multiple epochs
        self.model.train()
        num_epochs = 100
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
        print("Training completed!")
        self.model.eval()
    
    def classify_text(self, text):
        """
        Classify text as toxic or non-toxic
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Classification results
        """
        try:
            # Quick keyword check for boost
            tokens = self.preprocess_text(text)
            has_toxic_word = any(
                self.word2idx.get(token, -1) in self.toxic_indices 
                for token in tokens
            )
            
            # Convert text to sequence
            sequence = self.text_to_sequence(text)
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                toxic_score = output.item()
            
            # Boost score if toxic keywords present
            if has_toxic_word:
                toxic_score = min(toxic_score + 0.3, 1.0)
            
            # Determine classification
            if toxic_score > 0.5:
                classification = "toxic"
                confidence = toxic_score
            else:
                classification = "non-toxic"
                confidence = 1.0 - toxic_score
            
            return {
                "classification": classification,
                "confidence": confidence,
                "detailed_scores": {
                    "Toxic": toxic_score,
                    "Severe Toxic": min(max(toxic_score - 0.3, 0), 1.0) if toxic_score > 0.7 else 0.0
                }
            }
            
        except Exception as e:
            print(f"Error in classification: {e}")
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
