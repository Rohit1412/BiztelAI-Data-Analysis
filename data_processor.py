from abc import ABC, abstractmethod
import pandas as pd
import time
import logging
import functools
import json
import os
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessorDecorator:
    """Base decorator class for adding functionality to data processors."""
    
    def __init__(self, processor):
        self.processor = processor
        self.__class__.__name__ = processor.__class__.__name__
        functools.update_wrapper(self, processor)
        
    def __call__(self, data, **kwargs):
        return self.process(data, **kwargs)
    
    def process(self, data, **kwargs):
        return self.processor.process(data, **kwargs)
    
    def get_metadata(self):
        return self.processor.get_metadata()
    
    def validate(self, data):
        return self.processor.validate(data)

class TimingDecorator(ProcessorDecorator):
    """Decorator that measures execution time of data processing."""
    
    def process(self, data, **kwargs):
        start_time = time.time()
        result = self.processor.process(data, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{self.processor.__class__.__name__} execution time: {execution_time:.4f} seconds")
        
        # Update processor metadata with timing information
        metadata = self.processor.get_metadata()
        if 'timing' not in metadata:
            metadata['timing'] = {}
        metadata['timing']['execution_time'] = execution_time
        
        return result

class LoggingDecorator(ProcessorDecorator):
    """Decorator that logs information about data processing."""
    
    def process(self, data, **kwargs):
        logger.info(f"Starting {self.processor.__class__.__name__}")
        logger.info(f"Input data shape: {data.shape if hasattr(data, 'shape') else 'Not a DataFrame'}")
        
        result = self.processor.process(data, **kwargs)
        
        logger.info(f"Completed {self.processor.__class__.__name__}")
        if hasattr(result, 'shape'):
            logger.info(f"Output data shape: {result.shape}")
        
        return result

class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, name: str = None, config: Dict = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.metadata = {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config,
            'status': 'initialized'
        }
    
    def __call__(self, data, **kwargs):
        return self.process(data, **kwargs)
    
    @abstractmethod
    def process(self, data, **kwargs):
        """Process the input data and return the processed result."""
        pass
    
    def get_metadata(self) -> Dict:
        """Return metadata about the processor and its execution."""
        return self.metadata
    
    def add_metadata(self, key: str, value: Any):
        """Add or update metadata information."""
        self.metadata[key] = value
    
    def validate(self, data) -> bool:
        """Validate that the input data is in the expected format."""
        return True
    
    def with_timing(self):
        """Add timing measurement to the processor."""
        return TimingDecorator(self)
    
    def with_logging(self):
        """Add logging to the processor."""
        return LoggingDecorator(self)
    
    @classmethod
    def from_config(cls, config: Dict):
        """Create a processor instance from a configuration dictionary."""
        processor_name = config.get('name', cls.__name__)
        processor_config = config.get('config', {})
        processor = cls(name=processor_name, config=processor_config)
        
        # Apply decorators if specified in config
        if processor_config.get('timing', False):
            processor = processor.with_timing()
        if processor_config.get('logging', False):
            processor = processor.with_logging()
            
        return processor

class CompositeProcessor(DataProcessor):
    """A processor that combines multiple processors into a pipeline."""
    
    def __init__(self, name: str = None, config: Dict = None):
        super().__init__(name, config)
        self.processors = []
        self.metadata['processors'] = []
    
    def add_processor(self, processor: DataProcessor):
        """Add a processor to the pipeline."""
        self.processors.append(processor)
        self.metadata['processors'].append(processor.get_metadata())
        return self
    
    def process(self, data, **kwargs):
        """Process data through all processors in sequence."""
        result = data
        for processor in self.processors:
            if processor.validate(result):
                result = processor.process(result, **kwargs)
                self.add_metadata('last_processor', processor.name)
            else:
                logger.error(f"Validation failed for processor: {processor.name}")
                self.add_metadata('error', f"Validation failed at {processor.name}")
                break
        
        self.add_metadata('status', 'completed')
        return result
    
    def get_processor(self, name: str) -> Optional[DataProcessor]:
        """Get a processor by name."""
        for processor in self.processors:
            if processor.name == name:
                return processor
        return None

class ProcessorFactory:
    """Factory class for creating and configuring data processors."""
    
    @staticmethod
    def create_processor(processor_type: str, config: Dict) -> DataProcessor:
        """Create a processor instance based on its type and configuration."""
        from importlib import import_module
        
        processor_map = {
            # Register processor types here
            'composite': CompositeProcessor,
            'loader': 'DataLoader',
            'cleaner': 'DataCleaner',
            'text_processor': 'TextPreprocessor',
            'transformer': 'FeatureTransformer'
        }
        
        if processor_type in processor_map:
            processor_class = processor_map[processor_type]
            if isinstance(processor_class, str):
                # Need to import the class
                try:
                    module = import_module('data_pipeline')
                    processor_class = getattr(module, processor_class)
                except (ImportError, AttributeError) as e:
                    logger.error(f"Failed to import processor class {processor_class}: {e}")
                    raise ValueError(f"Could not import processor class: {processor_class}")
        else:
            # Attempt to dynamically import processor class
            try:
                module_name = config.get('module', 'data_pipeline')
                class_name = config.get('name')
                module = import_module(module_name)
                processor_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to import processor: {e}")
                raise ValueError(f"Unknown processor type: {processor_type}")
        
        return processor_class.from_config(config)
    
    @staticmethod
    def from_config_file(config_file: str) -> CompositeProcessor:
        """Create a composite processor (pipeline) from a JSON configuration file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Create pipeline
            pipeline_name = config.get('name', 'DataPipeline')
            pipeline_config = {
                'description': config.get('description', ''),
                'version': config.get('version', '1.0.0'),
                'input_file': config.get('input_file', ''),
                'output_file': config.get('output_file', ''),
                'logging': config.get('logging', {}),
                'caching': config.get('caching', {'enabled': False}),
                'parallelism': config.get('parallelism', {'enabled': False})
            }
            
            pipeline = CompositeProcessor(name=pipeline_name, config=pipeline_config)
            
            # Configure logging
            logging_config = config.get('logging', {})
            log_level = getattr(logging, logging_config.get('level', 'INFO'))
            log_file = logging_config.get('file')
            log_to_console = logging_config.get('console', True)
            
            # Set up logging
            handlers = []
            if log_to_console:
                console_handler = logging.StreamHandler()
                handlers.append(console_handler)
            
            if log_file:
                file_handler = logging.FileHandler(log_file)
                handlers.append(file_handler)
            
            if handlers:
                logger.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                for handler in handlers:
                    handler.setFormatter(formatter)
                    logger.addHandler(handler)
            
            # Add processors to pipeline
            for processor_config in config.get('processors', []):
                if not processor_config.get('enabled', True):
                    continue
                    
                processor_type = processor_config.get('type')
                processor = ProcessorFactory.create_processor(processor_type, processor_config)
                pipeline.add_processor(processor)
            
            return pipeline
        
        except Exception as e:
            logger.error(f"Error creating pipeline from config file: {e}")
            raise 