"""
Database Management Module
Handles CSV database operations for storing user inputs and classifications
"""

import csv
import os
from datetime import datetime
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path="toxic_content_database.csv"):
        """
        Initialize database manager
        
        Args:
            db_path (str): Path to CSV database file
        """
        self.db_path = db_path
        self.columns = [
            "timestamp",
            "input_type",
            "input_text",
            "classification",
            "confidence",
            "detailed_scores"
        ]
        
        # Create database file if it doesn't exist
        if not os.path.exists(self.db_path):
            self._create_database()
    
    def _create_database(self):
        """Create new CSV database with headers"""
        with open(self.db_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
        print(f"Database created: {self.db_path}")
    
    def add_record(self, input_type, input_text, classification_result):
        """
        Add a new record to the database
        
        Args:
            input_type (str): Type of input ('text' or 'image_caption')
            input_text (str): The input text or caption
            classification_result (dict): Classification results
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        record = [
            timestamp,
            input_type,
            input_text,
            classification_result.get("classification", "unknown"),
            classification_result.get("confidence", 0.0),
            str(classification_result.get("detailed_scores", {}))
        ]
        
        with open(self.db_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(record)
        
        print(f"Record added to database: {input_type}")
    
    def get_all_records(self):
        """
        Retrieve all records from the database
        
        Returns:
            pandas.DataFrame: All records
        """
        try:
            df = pd.read_csv(self.db_path)
            return df
        except Exception as e:
            print(f"Error reading database: {e}")
            return pd.DataFrame(columns=self.columns)
    
    def get_statistics(self):
        """
        Get statistics from the database
        
        Returns:
            dict: Statistics about classifications
        """
        df = self.get_all_records()
        
        if df.empty:
            return {
                "total_records": 0,
                "text_inputs": 0,
                "image_inputs": 0,
                "classifications": {}
            }
        
        stats = {
            "total_records": len(df),
            "text_inputs": len(df[df['input_type'] == 'text']),
            "image_inputs": len(df[df['input_type'] == 'image_caption']),
            "classifications": df['classification'].value_counts().to_dict()
        }
        
        return stats
    
    def clear_database(self):
        """Clear all records from the database"""
        self._create_database()
        print("Database cleared!")

# Singleton instance
_db_instance = None

def get_database():
    """Get or create DatabaseManager instance (singleton pattern)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
