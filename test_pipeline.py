<<<<<<< HEAD
from data_pipeline import DataPipeline
import pandas as pd
import os

def test_pipeline():
    """Test the data pipeline with a small sample of data."""
    
    # Create a small sample JSON file for testing
    sample_data = {
        "test_id_1": {
            "article_url": "https://www.example.com/article1",
            "config": "A",
            "content": [
                {
                    "message": "This is a test message from agent 1.",
                    "agent": "agent_1",
                    "sentiment": "Neutral",
                    "knowledge_source": ["FS1"],
                    "turn_rating": "Good"
                },
                {
                    "message": "I'm responding to the test message.",
                    "agent": "agent_2",
                    "sentiment": "Positive",
                    "knowledge_source": ["FS2"],
                    "turn_rating": "Good"
                }
            ]
        },
        "test_id_2": {
            "article_url": "https://www.example.com/article2",
            "config": "B",
            "content": [
                {
                    "message": "Another test message with missing data.",
                    "agent": "agent_1",
                    "sentiment": None,
                    "knowledge_source": [],
                    "turn_rating": "Bad"
                }
            ]
        }
    }
    
    # Save sample data to temp file
    import json
    with open("test_sample.json", "w") as f:
        json.dump(sample_data, f)
    
    # Run pipeline on test data
    print("Testing pipeline with sample data...")
    pipeline = DataPipeline("test_sample.json")
    result_df = pipeline.run()
    
    # Verify results
    if result_df is not None:
        print("\nTest Results:")
        print(f"Shape: {result_df.shape}")
        print(f"Columns: {result_df.columns.tolist()}")
        print("\nSample of processed data:")
        print(result_df.head())
        
        # Check that all expected steps were performed
        expected_columns = [
            'message_clean', 'tokens', 'tokens_no_stop', 'lemmatized',
            'agent_encoded', 'sentiment_encoded', 'message_length'
        ]
        
        missing_columns = [col for col in expected_columns if col not in result_df.columns]
        if missing_columns:
            print(f"\nWARNING: Missing expected columns: {missing_columns}")
        else:
            print("\nAll expected transformations were performed successfully!")
        
        # Clean up test file
        os.remove("test_sample.json")
        return True
    else:
        print("Pipeline test failed!")
        return False

if __name__ == "__main__":
=======
from data_pipeline import DataPipeline
import pandas as pd
import os

def test_pipeline():
    """Test the data pipeline with a small sample of data."""
    
    # Create a small sample JSON file for testing
    sample_data = {
        "test_id_1": {
            "article_url": "https://www.example.com/article1",
            "config": "A",
            "content": [
                {
                    "message": "This is a test message from agent 1.",
                    "agent": "agent_1",
                    "sentiment": "Neutral",
                    "knowledge_source": ["FS1"],
                    "turn_rating": "Good"
                },
                {
                    "message": "I'm responding to the test message.",
                    "agent": "agent_2",
                    "sentiment": "Positive",
                    "knowledge_source": ["FS2"],
                    "turn_rating": "Good"
                }
            ]
        },
        "test_id_2": {
            "article_url": "https://www.example.com/article2",
            "config": "B",
            "content": [
                {
                    "message": "Another test message with missing data.",
                    "agent": "agent_1",
                    "sentiment": None,
                    "knowledge_source": [],
                    "turn_rating": "Bad"
                }
            ]
        }
    }
    
    # Save sample data to temp file
    import json
    with open("test_sample.json", "w") as f:
        json.dump(sample_data, f)
    
    # Run pipeline on test data
    print("Testing pipeline with sample data...")
    pipeline = DataPipeline("test_sample.json")
    result_df = pipeline.run()
    
    # Verify results
    if result_df is not None:
        print("\nTest Results:")
        print(f"Shape: {result_df.shape}")
        print(f"Columns: {result_df.columns.tolist()}")
        print("\nSample of processed data:")
        print(result_df.head())
        
        # Check that all expected steps were performed
        expected_columns = [
            'message_clean', 'tokens', 'tokens_no_stop', 'lemmatized',
            'agent_encoded', 'sentiment_encoded', 'message_length'
        ]
        
        missing_columns = [col for col in expected_columns if col not in result_df.columns]
        if missing_columns:
            print(f"\nWARNING: Missing expected columns: {missing_columns}")
        else:
            print("\nAll expected transformations were performed successfully!")
        
        # Clean up test file
        os.remove("test_sample.json")
        return True
    else:
        print("Pipeline test failed!")
        return False

if __name__ == "__main__":
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    test_pipeline() 