<<<<<<< HEAD
import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache
import swifter  # For faster pandas apply operations
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from data_processor import DataProcessor, timing_decorator, log_step

# Download NLTK resources once at module level
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Cache the stopwords list for better performance
@lru_cache(maxsize=1)
def get_stopwords():
    return set(stopwords.words('english'))

# Initialize lemmatizer at module level
lemmatizer = WordNetLemmatizer()

class DataLoader(DataProcessor):
    """Class responsible for loading data from various file formats."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.file_path = config.get('file_path') if config else None
        self.chunk_size = config.get('chunk_size', 1000) if config else 1000
    
    def process(self, data, **kwargs):
        """Load data from file and return as DataFrame."""
        return self.load_json()
    
    def load_json(self) -> pd.DataFrame:
        """
        Load JSON file into a pandas DataFrame with optimized processing.
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            # For large files, consider using chunks with pd.read_json
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Pre-allocate lists for better performance
            conversations = []
            
            # Process the data in a more efficient way
            for conversation_id, conversation_data in data.items():
                article_url = conversation_data.get('article_url', '')
                config = conversation_data.get('config', '')
                
                # Batch process messages
                for message in conversation_data.get('content', []):
                    # Use dict comprehension instead of creating dict manually
                    knowledge_source = message.get('knowledge_source', [])
                    knowledge_str = ','.join(knowledge_source) if isinstance(knowledge_source, list) else ''
                    
                    message_data = {
                        'conversation_id': conversation_id,
                        'article_url': article_url,
                        'config': config,
                        'message': message.get('message', ''),
                        'agent': message.get('agent', ''),
                        'sentiment': message.get('sentiment', ''),
                        'knowledge_source': knowledge_str,
                        'turn_rating': message.get('turn_rating', '')
                    }
                    conversations.append(message_data)
            
            # Create DataFrame once instead of append operations
            df = pd.DataFrame(conversations)
            
            # Add metadata
            self.add_metadata('rows', df.shape[0])
            self.add_metadata('columns', df.shape[1])
            self.add_metadata('file_path', self.file_path)
            
            return df
        
        except Exception as e:
            logger = self.get_logger()
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_logger(self):
        """Get logger for this class."""
        import logging
        return logging.getLogger(self.name)


class DataCleaner(DataProcessor):
    """Class responsible for cleaning and preprocessing data."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.text_columns = config.get('text_columns', ['message', 'sentiment', 'knowledge_source']) if config else ['message', 'sentiment', 'knowledge_source']
        self.cat_columns = config.get('cat_columns', ['agent', 'config', 'turn_rating']) if config else ['agent', 'config', 'turn_rating']
        self.handle_missing = config.get('handle_missing', True) if config else True
        self.remove_duplicates = config.get('remove_duplicates', True) if config else True
        self.clean_text = config.get('clean_text', True) if config else True
        self.fix_data_types = config.get('fix_data_types', True) if config else True
    
    def process(self, data, **kwargs):
        """Clean the input data and return the cleaned result."""
        if not self.validate(data):
            return data
            
        result = data.copy()
        
        # Apply cleaning steps based on configuration
        if self.handle_missing:
            result = self.handle_missing_values(result)
        
        if self.remove_duplicates:
            result = self.remove_duplicates_fast(result)
        
        if self.clean_text:
            result = self.clean_text_data_vectorized(result)
        
        if self.fix_data_types:
            result = self.fix_data_types_optimized(result)
        
        return result
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with vectorized operations."""
        # Get missing value counts
        missing_counts = df.isna().sum()
        self.add_metadata('missing_values_before', missing_counts.to_dict())
        
        # Use vectorized operations instead of loops
        # For text columns: replace with empty string
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # For categorical columns: replace with 'Unknown'
        for col in self.cat_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Update metadata with after counts
        self.add_metadata('missing_values_after', df.isna().sum().to_dict())
        
        return df
    
    def remove_duplicates_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records efficiently."""
        initial_count = len(df)
        # Use keep='first' for consistent behavior and inplace=False for clarity
        df = df.drop_duplicates(keep='first')
        final_count = len(df)
        
        duplicates_removed = initial_count - final_count
        
        # Add metadata
        self.add_metadata('duplicates_removed', duplicates_removed)
        
        return df
    
    def clean_text_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text data using vectorized operations where possible."""
        if 'message' in df.columns:
            # Precompile regex for better performance
            url_pattern = re.compile(r'http\S+')
            special_char_pattern = re.compile(r'[^\w\s]')
            whitespace_pattern = re.compile(r'\s+')
            
            # Define a function to apply all cleaning in one pass
            def clean_text(text):
                text = str(text)
                text = url_pattern.sub('', text)
                text = special_char_pattern.sub(' ', text)
                text = whitespace_pattern.sub(' ', text).strip()
                return text
            
            # Use swifter for parallelized apply if dataset is large
            if len(df) > 10000:
                df['message_clean'] = df['message'].swifter.apply(clean_text)
            else:
                df['message_clean'] = df['message'].apply(clean_text)
            
            # Add metadata
            self.add_metadata('text_cleaned', True)
        
        return df
    
    def fix_data_types_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns have the correct data types, optimized for memory usage."""
        # Convert categorical columns in one pass
        categorical_data_before = df.memory_usage(deep=True).sum()
        
        # Only convert if the column exists
        cat_cols_to_convert = [col for col in self.cat_columns if col in df.columns]
        if cat_cols_to_convert:
            df[cat_cols_to_convert] = df[cat_cols_to_convert].astype('category')
        
        categorical_data_after = df.memory_usage(deep=True).sum()
        memory_savings = categorical_data_before - categorical_data_after
        
        # Add metadata
        self.add_metadata('data_types_fixed', True)
        self.add_metadata('memory_savings_bytes', memory_savings)
        
        return df


class TextPreprocessor(DataProcessor):
    """Class for text preprocessing tasks with optimized performance."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.text_column = config.get('text_column', 'message_clean') if config else 'message_clean'
        self.perform_tokenization = config.get('perform_tokenization', True) if config else True
        self.remove_stopwords = config.get('remove_stopwords', True) if config else True
        self.perform_lemmatization = config.get('perform_lemmatization', True) if config else True
        self.max_workers = config.get('max_workers', min(4, multiprocessing.cpu_count())) if config else min(4, multiprocessing.cpu_count())
    
    def process(self, data, **kwargs):
        """Preprocess text with parallel processing for better performance."""
        if not self.validate(data):
            return data
        
        result = data.copy()
        
        # Apply text preprocessing steps based on configuration
        if self.perform_tokenization and self.text_column in result.columns:
            result = self.tokenize_parallel(result)
        
        if self.remove_stopwords and 'tokens' in result.columns:
            result = self.remove_stopwords_parallel(result)
        
        if self.perform_lemmatization and 'tokens_no_stop' in result.columns:
            result = self.lemmatize_parallel(result)
        
        return result
    
    def tokenize_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tokenize text using parallel processing for large datasets."""
        if len(df) > 1000:
            # Use parallel processing for large datasets
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split into chunks for parallel processing
                chunks = np.array_split(df[self.text_column], self.max_workers)
                token_chunks = list(executor.map(self.tokenize_chunk, chunks))
                # Combine results
                tokens = []
                for chunk in token_chunks:
                    tokens.extend(chunk)
                df['tokens'] = tokens
        else:
            # For small datasets, use regular apply
            df['tokens'] = df[self.text_column].apply(word_tokenize)
        
        # Add metadata
        self.add_metadata('tokenized', True)
        self.add_metadata('tokens_count', df['tokens'].apply(len).sum())
        
        return df
    
    def tokenize_chunk(self, texts: pd.Series) -> List[List[str]]:
        """Tokenize a chunk of texts."""
        return [word_tokenize(str(text)) for text in texts]
    
    def remove_stopwords_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove stopwords using parallel processing for large datasets."""
        stop_words = get_stopwords()
        
        if len(df) > 1000:
            # Use parallel processing for large datasets
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split into chunks for parallel processing
                chunks = np.array_split(df['tokens'], self.max_workers)
                clean_chunks = list(executor.map(
                    lambda chunk: self.remove_stopwords_chunk(chunk, stop_words), 
                    chunks
                ))
                # Combine results
                clean_tokens = []
                for chunk in clean_chunks:
                    clean_tokens.extend(chunk)
                df['tokens_no_stop'] = clean_tokens
        else:
            # For small datasets, use regular apply
            df['tokens_no_stop'] = df['tokens'].apply(
                lambda tokens: [word for word in tokens if word.lower() not in stop_words]
            )
        
        # Add metadata
        self.add_metadata('stopwords_removed', True)
        self.add_metadata('tokens_after_stopword_removal', df['tokens_no_stop'].apply(len).sum())
        
        return df
    
    def remove_stopwords_chunk(self, tokens_series: pd.Series, stop_words: set) -> List[List[str]]:
        """Remove stopwords from a chunk of token lists."""
        return [[word for word in tokens if word.lower() not in stop_words] for tokens in tokens_series]
    
    def lemmatize_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lemmatize tokens using parallel processing for large datasets."""
        if len(df) > 1000:
            # Use parallel processing for large datasets
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split into chunks for parallel processing
                chunks = np.array_split(df['tokens_no_stop'], self.max_workers)
                lemma_chunks = list(executor.map(self.lemmatize_chunk, chunks))
                # Combine results
                lemmatized = []
                for chunk in lemma_chunks:
                    lemmatized.extend(chunk)
                df['lemmatized'] = lemmatized
        else:
            # For small datasets, use regular apply
            df['lemmatized'] = df['tokens_no_stop'].apply(
                lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
            )
        
        # Add metadata
        self.add_metadata('lemmatized', True)
        
        return df
    
    def lemmatize_chunk(self, tokens_series: pd.Series) -> List[List[str]]:
        """Lemmatize a chunk of token lists."""
        return [[lemmatizer.lemmatize(word) for word in tokens] for tokens in tokens_series]


class FeatureTransformer(DataProcessor):
    """Class for feature transformation with optimized performance."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.categorical_columns = config.get('categorical_columns', ['agent', 'config', 'turn_rating', 'sentiment']) if config else ['agent', 'config', 'turn_rating', 'sentiment']
        self.create_length_feature = config.get('create_length_feature', True) if config else True
        self.create_word_count_feature = config.get('create_word_count_feature', True) if config else True
        self.label_encoders = {}
    
    def process(self, data, **kwargs):
        """Transform features with optimized operations."""
        if not self.validate(data):
            return data
        
        result = data.copy()
        
        # Apply column filtering to work only with existing columns
        cat_columns_present = [col for col in self.categorical_columns if col in result.columns]
        
        # Encode categorical variables
        if cat_columns_present:
            result = self.encode_categorical_optimized(result, cat_columns_present)
        
        # Create additional features
        if self.create_length_feature and 'message' in result.columns:
            # Vectorized operation instead of apply
            result['message_length'] = result['message'].str.len()
            self.add_metadata('feature_message_length', True)
        
        if self.create_word_count_feature and 'tokens' in result.columns:
            # Vectorized operation is faster than apply for simple length calculation
            result['word_count'] = result['tokens'].str.len()
            self.add_metadata('feature_word_count', True)
        
        return result
    
    def encode_categorical_optimized(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables with optimized memory usage."""
        for col in columns:
            # Create encoder
            encoder = LabelEncoder()
            
            # Get only unique values to speed up fit_transform
            unique_values = df[col].unique()
            unique_encoded = encoder.fit_transform(unique_values)
            
            # Create mapping for faster transform
            value_to_encoding = dict(zip(unique_values, unique_encoded))
            
            # Use map which is faster than transform for the whole column
            df[f'{col}_encoded'] = df[col].map(value_to_encoding)
            
            # Store encoder for future use
            self.label_encoders[col] = encoder
            
            # Create mapping for reference
            mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            
            # Add metadata
            self.add_metadata(f'{col}_encoded', True)
            self.add_metadata(f'{col}_mapping', mapping)
        
        return df


class DataPipeline(DataProcessor):
    """Main class to orchestrate the data pipeline with parallel execution."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.parallel = config.get('parallelism', {}).get('enabled', False) if config else False
        self.max_workers = config.get('parallelism', {}).get('max_workers', multiprocessing.cpu_count()) if config else multiprocessing.cpu_count()
    
    def process(self, data, **kwargs):
        """Process data through individual processors or in parallel."""
        if not hasattr(self, 'processors') or not self.processors:
            return data
        
        result = data
        
        # Check if we should process in parallel
        if self.parallel and len(self.processors) > 1:
            try:
                # Split processors into independent groups that can run in parallel
                # For example, data loading must be done first, but cleaning and feature
                # engineering might be parallelizable
                
                # For simplicity, we'll process the first step (loading) separately
                # and then parallelize the rest if possible
                first_processor = self.processors[0]
                result = first_processor.process(result)
                
                # Process remaining steps in parallel if applicable
                remaining_processors = self.processors[1:]
                
                if len(remaining_processors) > 1:
                    # Split the data into chunks for parallel processing
                    num_chunks = min(len(remaining_processors), self.max_workers)
                    
                    # Only split if data is a DataFrame with enough rows
                    if isinstance(result, pd.DataFrame) and len(result) > 1000:
                        data_chunks = np.array_split(result, num_chunks)
                        
                        # Create processing tasks
                        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                            processed_chunks = []
                            
                            # Process each chunk through all processors
                            for chunk in data_chunks:
                                # Process chunk through all processors
                                future = executor.submit(self.process_chunk, chunk, remaining_processors)
                                processed_chunks.append(future)
                            
                            # Collect results
                            processed_results = [future.result() for future in processed_chunks]
                            
                            # Combine results
                            result = pd.concat(processed_results, ignore_index=True)
                    else:
                        # If data can't be split, process sequentially
                        for processor in remaining_processors:
                            result = processor.process(result)
                else:
                    # If only one remaining processor, just process normally
                    for processor in remaining_processors:
                        result = processor.process(result)
            except Exception as e:
                # Log error and fall back to sequential processing
                import logging
                logger = logging.getLogger(self.name)
                logger.error(f"Error in parallel processing: {str(e)}. Falling back to sequential processing.")
                
                # Process sequentially
                result = self.process_sequential(data)
        else:
            # Process sequentially
            result = self.process_sequential(data)
        
        return result
    
    def process_sequential(self, data):
        """Process data sequentially through all processors."""
        result = data
        for processor in self.processors:
            if processor.validate(result):
                result = processor.process(result)
                self.add_metadata('last_processor', processor.name)
            else:
                import logging
                logger = logging.getLogger(self.name)
                logger.error(f"Validation failed for processor: {processor.name}")
                self.add_metadata('error', f"Validation failed at {processor.name}")
                break
        
        self.add_metadata('status', 'completed')
        return result
    
    @staticmethod
    def process_chunk(chunk, processors):
        """Process a data chunk through a list of processors."""
        result = chunk
        for processor in processors:
            if processor.validate(result):
                result = processor.process(result)
        return result
    
    def run_optimized(self, input_file: str, output_file: str) -> pd.DataFrame:
        """Run the optimized data pipeline."""
        # Initialize with empty DataFrame (will be ignored by loader)
        result = self.process(pd.DataFrame())
        
        if result is not None and not isinstance(result, pd.DataFrame):
            import logging
            logger = logging.getLogger(self.name)
            logger.error("Failed to process data. Pipeline terminated.")
            return None
            
        # Save result if output file is provided
        if output_file and not result.empty:
            # Determine file format based on extension
            extension = output_file.split('.')[-1].lower()
            
            if extension == 'csv':
                result.to_csv(output_file, index=False)
            elif extension == 'parquet':
                result.to_parquet(output_file, index=False)
            elif extension in ['pkl', 'pickle']:
                result.to_pickle(output_file)
            elif extension == 'json':
                result.to_json(output_file, orient='records', lines=True)
            else:
                # Default to CSV
                result.to_csv(output_file, index=False)
        
        return result


def run_pipeline(input_path: str, output_path: str, config_path: str = None) -> pd.DataFrame:
    """
    Run the complete data pipeline and save results.
    
    Args:
        input_path: Path to input data file
        output_path: Path where to save the processed data
        config_path: Optional path to configuration file
        
    Returns:
        Processed DataFrame
    """
    from data_processor import ProcessorFactory
    
    if config_path:
        # Load from configuration file
        pipeline = ProcessorFactory.from_config_file(config_path)
    else:
        # Create default pipeline
        pipeline = DataPipeline(config={
            'name': 'BiztelAI_DataPipeline',
            'parallelism': {'enabled': True, 'max_workers': multiprocessing.cpu_count()}
        })
        
        # Add processors
        from data_processor import CompositeProcessor
        if isinstance(pipeline, CompositeProcessor):
            # Add default processors
            loader = DataLoader(config={'file_path': input_path})
            cleaner = DataCleaner()
            text_processor = TextPreprocessor()
            transformer = FeatureTransformer()
            
            pipeline.add_processor(loader)
            pipeline.add_processor(cleaner)
            pipeline.add_processor(text_processor)
            pipeline.add_processor(transformer)
    
    # Run pipeline
    df = pipeline.run_optimized(input_path, output_path)
    
    return df


if __name__ == "__main__":
    input_file = "BiztelAI_DS_Dataset_Mar'25.json"
    output_file = "processed_data.csv"
    
=======
import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache
import swifter  # For faster pandas apply operations
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from data_processor import DataProcessor, timing_decorator, log_step

# Download NLTK resources once at module level
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Cache the stopwords list for better performance
@lru_cache(maxsize=1)
def get_stopwords():
    return set(stopwords.words('english'))

# Initialize lemmatizer at module level
lemmatizer = WordNetLemmatizer()

class DataLoader(DataProcessor):
    """Class responsible for loading data from various file formats."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.file_path = config.get('file_path') if config else None
        self.chunk_size = config.get('chunk_size', 1000) if config else 1000
    
    def process(self, data, **kwargs):
        """Load data from file and return as DataFrame."""
        return self.load_json()
    
    def load_json(self) -> pd.DataFrame:
        """
        Load JSON file into a pandas DataFrame with optimized processing.
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            # For large files, consider using chunks with pd.read_json
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Pre-allocate lists for better performance
            conversations = []
            
            # Process the data in a more efficient way
            for conversation_id, conversation_data in data.items():
                article_url = conversation_data.get('article_url', '')
                config = conversation_data.get('config', '')
                
                # Batch process messages
                for message in conversation_data.get('content', []):
                    # Use dict comprehension instead of creating dict manually
                    knowledge_source = message.get('knowledge_source', [])
                    knowledge_str = ','.join(knowledge_source) if isinstance(knowledge_source, list) else ''
                    
                    message_data = {
                        'conversation_id': conversation_id,
                        'article_url': article_url,
                        'config': config,
                        'message': message.get('message', ''),
                        'agent': message.get('agent', ''),
                        'sentiment': message.get('sentiment', ''),
                        'knowledge_source': knowledge_str,
                        'turn_rating': message.get('turn_rating', '')
                    }
                    conversations.append(message_data)
            
            # Create DataFrame once instead of append operations
            df = pd.DataFrame(conversations)
            
            # Add metadata
            self.add_metadata('rows', df.shape[0])
            self.add_metadata('columns', df.shape[1])
            self.add_metadata('file_path', self.file_path)
            
            return df
        
        except Exception as e:
            logger = self.get_logger()
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_logger(self):
        """Get logger for this class."""
        import logging
        return logging.getLogger(self.name)


class DataCleaner(DataProcessor):
    """Class responsible for cleaning and preprocessing data."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.text_columns = config.get('text_columns', ['message', 'sentiment', 'knowledge_source']) if config else ['message', 'sentiment', 'knowledge_source']
        self.cat_columns = config.get('cat_columns', ['agent', 'config', 'turn_rating']) if config else ['agent', 'config', 'turn_rating']
        self.handle_missing = config.get('handle_missing', True) if config else True
        self.remove_duplicates = config.get('remove_duplicates', True) if config else True
        self.clean_text = config.get('clean_text', True) if config else True
        self.fix_data_types = config.get('fix_data_types', True) if config else True
    
    def process(self, data, **kwargs):
        """Clean the input data and return the cleaned result."""
        if not self.validate(data):
            return data
            
        result = data.copy()
        
        # Apply cleaning steps based on configuration
        if self.handle_missing:
            result = self.handle_missing_values(result)
        
        if self.remove_duplicates:
            result = self.remove_duplicates_fast(result)
        
        if self.clean_text:
            result = self.clean_text_data_vectorized(result)
        
        if self.fix_data_types:
            result = self.fix_data_types_optimized(result)
        
        return result
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with vectorized operations."""
        # Get missing value counts
        missing_counts = df.isna().sum()
        self.add_metadata('missing_values_before', missing_counts.to_dict())
        
        # Use vectorized operations instead of loops
        # For text columns: replace with empty string
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # For categorical columns: replace with 'Unknown'
        for col in self.cat_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Update metadata with after counts
        self.add_metadata('missing_values_after', df.isna().sum().to_dict())
        
        return df
    
    def remove_duplicates_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records efficiently."""
        initial_count = len(df)
        # Use keep='first' for consistent behavior and inplace=False for clarity
        df = df.drop_duplicates(keep='first')
        final_count = len(df)
        
        duplicates_removed = initial_count - final_count
        
        # Add metadata
        self.add_metadata('duplicates_removed', duplicates_removed)
        
        return df
    
    def clean_text_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text data using vectorized operations where possible."""
        if 'message' in df.columns:
            # Precompile regex for better performance
            url_pattern = re.compile(r'http\S+')
            special_char_pattern = re.compile(r'[^\w\s]')
            whitespace_pattern = re.compile(r'\s+')
            
            # Define a function to apply all cleaning in one pass
            def clean_text(text):
                text = str(text)
                text = url_pattern.sub('', text)
                text = special_char_pattern.sub(' ', text)
                text = whitespace_pattern.sub(' ', text).strip()
                return text
            
            # Use swifter for parallelized apply if dataset is large
            if len(df) > 10000:
                df['message_clean'] = df['message'].swifter.apply(clean_text)
            else:
                df['message_clean'] = df['message'].apply(clean_text)
            
            # Add metadata
            self.add_metadata('text_cleaned', True)
        
        return df
    
    def fix_data_types_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns have the correct data types, optimized for memory usage."""
        # Convert categorical columns in one pass
        categorical_data_before = df.memory_usage(deep=True).sum()
        
        # Only convert if the column exists
        cat_cols_to_convert = [col for col in self.cat_columns if col in df.columns]
        if cat_cols_to_convert:
            df[cat_cols_to_convert] = df[cat_cols_to_convert].astype('category')
        
        categorical_data_after = df.memory_usage(deep=True).sum()
        memory_savings = categorical_data_before - categorical_data_after
        
        # Add metadata
        self.add_metadata('data_types_fixed', True)
        self.add_metadata('memory_savings_bytes', memory_savings)
        
        return df


class TextPreprocessor(DataProcessor):
    """Class for text preprocessing tasks with optimized performance."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.text_column = config.get('text_column', 'message_clean') if config else 'message_clean'
        self.perform_tokenization = config.get('perform_tokenization', True) if config else True
        self.remove_stopwords = config.get('remove_stopwords', True) if config else True
        self.perform_lemmatization = config.get('perform_lemmatization', True) if config else True
        self.max_workers = config.get('max_workers', min(4, multiprocessing.cpu_count())) if config else min(4, multiprocessing.cpu_count())
    
    def process(self, data, **kwargs):
        """Preprocess text with parallel processing for better performance."""
        if not self.validate(data):
            return data
        
        result = data.copy()
        
        # Apply text preprocessing steps based on configuration
        if self.perform_tokenization and self.text_column in result.columns:
            result = self.tokenize_parallel(result)
        
        if self.remove_stopwords and 'tokens' in result.columns:
            result = self.remove_stopwords_parallel(result)
        
        if self.perform_lemmatization and 'tokens_no_stop' in result.columns:
            result = self.lemmatize_parallel(result)
        
        return result
    
    def tokenize_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tokenize text using parallel processing for large datasets."""
        if len(df) > 1000:
            # Use parallel processing for large datasets
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split into chunks for parallel processing
                chunks = np.array_split(df[self.text_column], self.max_workers)
                token_chunks = list(executor.map(self.tokenize_chunk, chunks))
                # Combine results
                tokens = []
                for chunk in token_chunks:
                    tokens.extend(chunk)
                df['tokens'] = tokens
        else:
            # For small datasets, use regular apply
            df['tokens'] = df[self.text_column].apply(word_tokenize)
        
        # Add metadata
        self.add_metadata('tokenized', True)
        self.add_metadata('tokens_count', df['tokens'].apply(len).sum())
        
        return df
    
    def tokenize_chunk(self, texts: pd.Series) -> List[List[str]]:
        """Tokenize a chunk of texts."""
        return [word_tokenize(str(text)) for text in texts]
    
    def remove_stopwords_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove stopwords using parallel processing for large datasets."""
        stop_words = get_stopwords()
        
        if len(df) > 1000:
            # Use parallel processing for large datasets
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split into chunks for parallel processing
                chunks = np.array_split(df['tokens'], self.max_workers)
                clean_chunks = list(executor.map(
                    lambda chunk: self.remove_stopwords_chunk(chunk, stop_words), 
                    chunks
                ))
                # Combine results
                clean_tokens = []
                for chunk in clean_chunks:
                    clean_tokens.extend(chunk)
                df['tokens_no_stop'] = clean_tokens
        else:
            # For small datasets, use regular apply
            df['tokens_no_stop'] = df['tokens'].apply(
                lambda tokens: [word for word in tokens if word.lower() not in stop_words]
            )
        
        # Add metadata
        self.add_metadata('stopwords_removed', True)
        self.add_metadata('tokens_after_stopword_removal', df['tokens_no_stop'].apply(len).sum())
        
        return df
    
    def remove_stopwords_chunk(self, tokens_series: pd.Series, stop_words: set) -> List[List[str]]:
        """Remove stopwords from a chunk of token lists."""
        return [[word for word in tokens if word.lower() not in stop_words] for tokens in tokens_series]
    
    def lemmatize_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lemmatize tokens using parallel processing for large datasets."""
        if len(df) > 1000:
            # Use parallel processing for large datasets
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split into chunks for parallel processing
                chunks = np.array_split(df['tokens_no_stop'], self.max_workers)
                lemma_chunks = list(executor.map(self.lemmatize_chunk, chunks))
                # Combine results
                lemmatized = []
                for chunk in lemma_chunks:
                    lemmatized.extend(chunk)
                df['lemmatized'] = lemmatized
        else:
            # For small datasets, use regular apply
            df['lemmatized'] = df['tokens_no_stop'].apply(
                lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
            )
        
        # Add metadata
        self.add_metadata('lemmatized', True)
        
        return df
    
    def lemmatize_chunk(self, tokens_series: pd.Series) -> List[List[str]]:
        """Lemmatize a chunk of token lists."""
        return [[lemmatizer.lemmatize(word) for word in tokens] for tokens in tokens_series]


class FeatureTransformer(DataProcessor):
    """Class for feature transformation with optimized performance."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.categorical_columns = config.get('categorical_columns', ['agent', 'config', 'turn_rating', 'sentiment']) if config else ['agent', 'config', 'turn_rating', 'sentiment']
        self.create_length_feature = config.get('create_length_feature', True) if config else True
        self.create_word_count_feature = config.get('create_word_count_feature', True) if config else True
        self.label_encoders = {}
    
    def process(self, data, **kwargs):
        """Transform features with optimized operations."""
        if not self.validate(data):
            return data
        
        result = data.copy()
        
        # Apply column filtering to work only with existing columns
        cat_columns_present = [col for col in self.categorical_columns if col in result.columns]
        
        # Encode categorical variables
        if cat_columns_present:
            result = self.encode_categorical_optimized(result, cat_columns_present)
        
        # Create additional features
        if self.create_length_feature and 'message' in result.columns:
            # Vectorized operation instead of apply
            result['message_length'] = result['message'].str.len()
            self.add_metadata('feature_message_length', True)
        
        if self.create_word_count_feature and 'tokens' in result.columns:
            # Vectorized operation is faster than apply for simple length calculation
            result['word_count'] = result['tokens'].str.len()
            self.add_metadata('feature_word_count', True)
        
        return result
    
    def encode_categorical_optimized(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables with optimized memory usage."""
        for col in columns:
            # Create encoder
            encoder = LabelEncoder()
            
            # Get only unique values to speed up fit_transform
            unique_values = df[col].unique()
            unique_encoded = encoder.fit_transform(unique_values)
            
            # Create mapping for faster transform
            value_to_encoding = dict(zip(unique_values, unique_encoded))
            
            # Use map which is faster than transform for the whole column
            df[f'{col}_encoded'] = df[col].map(value_to_encoding)
            
            # Store encoder for future use
            self.label_encoders[col] = encoder
            
            # Create mapping for reference
            mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            
            # Add metadata
            self.add_metadata(f'{col}_encoded', True)
            self.add_metadata(f'{col}_mapping', mapping)
        
        return df


class DataPipeline(DataProcessor):
    """Main class to orchestrate the data pipeline with parallel execution."""
    
    def __init__(self, name: str = None, config: Dict = None):
        """Initialize with configuration."""
        super().__init__(name, config)
        self.parallel = config.get('parallelism', {}).get('enabled', False) if config else False
        self.max_workers = config.get('parallelism', {}).get('max_workers', multiprocessing.cpu_count()) if config else multiprocessing.cpu_count()
    
    def process(self, data, **kwargs):
        """Process data through individual processors or in parallel."""
        if not hasattr(self, 'processors') or not self.processors:
            return data
        
        result = data
        
        # Check if we should process in parallel
        if self.parallel and len(self.processors) > 1:
            try:
                # Split processors into independent groups that can run in parallel
                # For example, data loading must be done first, but cleaning and feature
                # engineering might be parallelizable
                
                # For simplicity, we'll process the first step (loading) separately
                # and then parallelize the rest if possible
                first_processor = self.processors[0]
                result = first_processor.process(result)
                
                # Process remaining steps in parallel if applicable
                remaining_processors = self.processors[1:]
                
                if len(remaining_processors) > 1:
                    # Split the data into chunks for parallel processing
                    num_chunks = min(len(remaining_processors), self.max_workers)
                    
                    # Only split if data is a DataFrame with enough rows
                    if isinstance(result, pd.DataFrame) and len(result) > 1000:
                        data_chunks = np.array_split(result, num_chunks)
                        
                        # Create processing tasks
                        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                            processed_chunks = []
                            
                            # Process each chunk through all processors
                            for chunk in data_chunks:
                                # Process chunk through all processors
                                future = executor.submit(self.process_chunk, chunk, remaining_processors)
                                processed_chunks.append(future)
                            
                            # Collect results
                            processed_results = [future.result() for future in processed_chunks]
                            
                            # Combine results
                            result = pd.concat(processed_results, ignore_index=True)
                    else:
                        # If data can't be split, process sequentially
                        for processor in remaining_processors:
                            result = processor.process(result)
                else:
                    # If only one remaining processor, just process normally
                    for processor in remaining_processors:
                        result = processor.process(result)
            except Exception as e:
                # Log error and fall back to sequential processing
                import logging
                logger = logging.getLogger(self.name)
                logger.error(f"Error in parallel processing: {str(e)}. Falling back to sequential processing.")
                
                # Process sequentially
                result = self.process_sequential(data)
        else:
            # Process sequentially
            result = self.process_sequential(data)
        
        return result
    
    def process_sequential(self, data):
        """Process data sequentially through all processors."""
        result = data
        for processor in self.processors:
            if processor.validate(result):
                result = processor.process(result)
                self.add_metadata('last_processor', processor.name)
            else:
                import logging
                logger = logging.getLogger(self.name)
                logger.error(f"Validation failed for processor: {processor.name}")
                self.add_metadata('error', f"Validation failed at {processor.name}")
                break
        
        self.add_metadata('status', 'completed')
        return result
    
    @staticmethod
    def process_chunk(chunk, processors):
        """Process a data chunk through a list of processors."""
        result = chunk
        for processor in processors:
            if processor.validate(result):
                result = processor.process(result)
        return result
    
    def run_optimized(self, input_file: str, output_file: str) -> pd.DataFrame:
        """Run the optimized data pipeline."""
        # Initialize with empty DataFrame (will be ignored by loader)
        result = self.process(pd.DataFrame())
        
        if result is not None and not isinstance(result, pd.DataFrame):
            import logging
            logger = logging.getLogger(self.name)
            logger.error("Failed to process data. Pipeline terminated.")
            return None
            
        # Save result if output file is provided
        if output_file and not result.empty:
            # Determine file format based on extension
            extension = output_file.split('.')[-1].lower()
            
            if extension == 'csv':
                result.to_csv(output_file, index=False)
            elif extension == 'parquet':
                result.to_parquet(output_file, index=False)
            elif extension in ['pkl', 'pickle']:
                result.to_pickle(output_file)
            elif extension == 'json':
                result.to_json(output_file, orient='records', lines=True)
            else:
                # Default to CSV
                result.to_csv(output_file, index=False)
        
        return result


def run_pipeline(input_path: str, output_path: str, config_path: str = None) -> pd.DataFrame:
    """
    Run the complete data pipeline and save results.
    
    Args:
        input_path: Path to input data file
        output_path: Path where to save the processed data
        config_path: Optional path to configuration file
        
    Returns:
        Processed DataFrame
    """
    from data_processor import ProcessorFactory
    
    if config_path:
        # Load from configuration file
        pipeline = ProcessorFactory.from_config_file(config_path)
    else:
        # Create default pipeline
        pipeline = DataPipeline(config={
            'name': 'BiztelAI_DataPipeline',
            'parallelism': {'enabled': True, 'max_workers': multiprocessing.cpu_count()}
        })
        
        # Add processors
        from data_processor import CompositeProcessor
        if isinstance(pipeline, CompositeProcessor):
            # Add default processors
            loader = DataLoader(config={'file_path': input_path})
            cleaner = DataCleaner()
            text_processor = TextPreprocessor()
            transformer = FeatureTransformer()
            
            pipeline.add_processor(loader)
            pipeline.add_processor(cleaner)
            pipeline.add_processor(text_processor)
            pipeline.add_processor(transformer)
    
    # Run pipeline
    df = pipeline.run_optimized(input_path, output_path)
    
    return df


if __name__ == "__main__":
    input_file = "BiztelAI_DS_Dataset_Mar'25.json"
    output_file = "processed_data.csv"
    
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    df = run_pipeline(input_file, output_file) 