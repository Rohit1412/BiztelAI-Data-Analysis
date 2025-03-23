#!/usr/bin/env python
"""
Test script for OOP Data Pipeline implementation
"""
import os
import unittest
import pandas as pd
import numpy as np
import json
import tempfile
from typing import Dict, Any

from data_processor import (
    DataProcessor, 
    CompositeProcessor, 
    ProcessorFactory,
    TimingDecorator,
    LoggingDecorator
)

# Create a sample dataset for testing
def create_test_data() -> pd.DataFrame:
    """Create a small test dataset with features similar to the BiztelAI dataset."""
    data = {
        'conversation_id': ['conv1', 'conv1', 'conv2', 'conv2', 'conv2'],
        'turn_id': [1, 2, 1, 2, 3],
        'agent': ['agent_1', 'agent_2', 'agent_1', 'agent_2', 'agent_1'],
        'message': [
            'Have you read the article about sports?',
            'Yes, it was interesting.',
            'What did you think of the political news?',
            'I found it controversial.',
            'Let\'s discuss something else.'
        ],
        'sentiment': [
            'Curious to dive deeper', 
            'Positive', 
            'Neutral', 
            'Slightly negative',
            'Neutral'
        ],
        'knowledge_source': [
            'https://example.com/sports', 
            '', 
            'https://example.com/politics',
            '',
            ''
        ],
        'turn_rating': ['Good', 'Good', 'Excellent', 'Good', 'Fair']
    }
    return pd.DataFrame(data)

# Concrete implementations for testing
class TestLoader(DataProcessor):
    """Test data loader for in-memory data."""
    
    def __init__(self, name: str = None, config: Dict = None):
        super().__init__(name, config)
        self.test_data = create_test_data()
    
    def process(self, data, **kwargs):
        """Returns the test data instead of loading from a file."""
        self.add_metadata('rows', len(self.test_data))
        self.add_metadata('columns', list(self.test_data.columns))
        return self.test_data

class TestProcessor(DataProcessor):
    """Simple processor that adds a column to the data."""
    
    def __init__(self, name: str = None, config: Dict = None):
        super().__init__(name, config)
        self.column_name = config.get('column_name', 'test_column')
        self.value = config.get('value', 1)
    
    def process(self, data, **kwargs):
        """Add a test column with a constant value."""
        result = data.copy()
        result[self.column_name] = self.value
        self.add_metadata('added_column', self.column_name)
        return result

class TestPipeline(unittest.TestCase):
    """Test cases for the data pipeline implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.config_file = os.path.join(tempfile.gettempdir(), 'test_config.json')
        self.config = {
            "name": "TestPipeline",
            "description": "Pipeline for testing",
            "logging": {
                "level": "INFO",
                "console": True
            },
            "processors": [
                {
                    "type": "test_loader",
                    "name": "TestLoader",
                    "enabled": True,
                    "config": {
                        "timing": True,
                        "logging": True
                    }
                },
                {
                    "type": "test_processor",
                    "name": "TestProcessor1",
                    "enabled": True,
                    "config": {
                        "timing": True,
                        "logging": True,
                        "column_name": "processed_flag",
                        "value": 1
                    }
                },
                {
                    "type": "test_processor",
                    "name": "TestProcessor2",
                    "enabled": True,
                    "config": {
                        "column_name": "score",
                        "value": 100
                    }
                }
            ]
        }
        
        # Write config to file
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)
        
        # Register test processors
        self.original_create_processor = ProcessorFactory.create_processor
        
        def mock_create_processor(processor_type, config):
            if processor_type == 'test_loader':
                return TestLoader.from_config(config)
            elif processor_type == 'test_processor':
                return TestProcessor.from_config(config)
            else:
                return self.original_create_processor(processor_type, config)
                
        ProcessorFactory.create_processor = mock_create_processor
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary config file
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        
        # Restore original method
        ProcessorFactory.create_processor = self.original_create_processor
    
    def test_processor_base_class(self):
        """Test DataProcessor base class functionality."""
        # Cannot instantiate abstract class directly
        with self.assertRaises(TypeError):
            dp = DataProcessor()
    
    def test_processor_decorators(self):
        """Test decorator functionality."""
        # Create a processor
        processor = TestProcessor(config={'column_name': 'test', 'value': 5})
        
        # Apply decorators
        processor_with_timing = TimingDecorator(processor)
        processor_with_timing_and_logging = LoggingDecorator(processor_with_timing)
        
        # Process data through decorated processor
        data = create_test_data()
        result = processor_with_timing_and_logging.process(data)
        
        # Check result
        self.assertIn('test', result.columns)
        self.assertTrue((result['test'] == 5).all())
        
        # Metadata should be accessible through decorators
        metadata = processor_with_timing_and_logging.get_metadata()
        self.assertEqual(metadata['added_column'], 'test')
    
    def test_composite_processor(self):
        """Test CompositeProcessor functionality."""
        # Create a composite processor
        pipeline = CompositeProcessor(name="TestComposite")
        
        # Add processors
        pipeline.add_processor(TestLoader())
        pipeline.add_processor(TestProcessor(config={'column_name': 'flag', 'value': 1}))
        pipeline.add_processor(TestProcessor(config={'column_name': 'score', 'value': 10}))
        
        # Process data
        result = pipeline.process(None)  # Initial data is None as the loader will create it
        
        # Check result
        self.assertIn('flag', result.columns)
        self.assertIn('score', result.columns)
        self.assertEqual(len(result), 5)  # Should have 5 rows from test data
    
    def test_factory_from_config(self):
        """Test creating a pipeline from configuration."""
        # Create pipeline from config file
        pipeline = ProcessorFactory.from_config_file(self.config_file)
        
        # Check pipeline
        self.assertIsInstance(pipeline, CompositeProcessor)
        self.assertEqual(pipeline.name, "TestPipeline")
        self.assertEqual(len(pipeline.processors), 3)
        
        # Process data
        result = pipeline.process(None)
        
        # Check result
        self.assertIn('processed_flag', result.columns)
        self.assertIn('score', result.columns)
        self.assertTrue((result['processed_flag'] == 1).all())
        self.assertTrue((result['score'] == 100).all())
        
        # Check metadata
        metadata = pipeline.get_metadata()
        self.assertEqual(metadata['status'], 'completed')
        self.assertGreaterEqual(len(metadata['processors']), 3)

if __name__ == '__main__':
    unittest.main() 