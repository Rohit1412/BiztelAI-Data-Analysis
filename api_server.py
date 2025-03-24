from flask import Flask, request, jsonify
import pandas as pd
import json
import os
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
from datetime import datetime
import uuid
from werkzeug.middleware.profiler import ProfilerMiddleware
from functools import wraps
from dotenv import load_dotenv
import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Any, Optional, Union
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from concurrent.futures import ThreadPoolExecutor
import threading
import tempfile
from cachetools import TTLCache, cached

# Load environment variables
load_dotenv()

# Import our data processing components
from data_pipeline import DataLoader, DataCleaner, TextPreprocessor, FeatureTransformer, DataPipeline
from transcript_analyzer import TranscriptAnalyzer, analyze_chat_transcript

# Setup Flask application
app = Flask(__name__)
CORS(app)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

# Get log level from environment
log_level_str = os.getenv('LOG_LEVEL', 'INFO')
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

# Get log file size and backup count from environment
log_file_size = int(os.getenv('LOG_FILE_SIZE', 10485760))  # Default: 10MB
log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', 5))   # Default: 5 files

# Create a rotating file handler
file_handler = RotatingFileHandler(
    'logs/api.log',
    maxBytes=log_file_size,
    backupCount=log_backup_count
)
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))

# Add console handler for development environment
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure app logger
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(log_level)

# Request tracking and performance metrics
request_metrics = {}

def track_request_time(f):
    """Decorator to track request processing time for analytics."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        request_id = str(uuid.uuid4())
        request.request_id = request_id
        
        # Log request details
        app.logger.info(f"Request {request_id} started: {request.method} {request.path} {request.remote_addr}")
        
        # Track request time
        start_time = time.time()
        response = f(*args, **kwargs)
        end_time = time.time()
        
        # Calculate duration
        duration = end_time - start_time
        
        # Store metrics
        request_metrics[request_id] = {
            'endpoint': request.path,
            'method': request.method,
            'duration': duration,
            'status_code': response[1] if isinstance(response, tuple) else 200,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log completion
        app.logger.info(f"Request {request_id} completed in {duration:.4f}s with status {request_metrics[request_id]['status_code']}")
        
        return response
    return decorated_function

# Cache for dataset summary to improve performance
dataset_summary_cache = None
cache_timestamp = None
CACHE_EXPIRY = int(os.getenv('CACHE_EXPIRY', 3600))  # Cache expiry in seconds from env

# Global variables to reduce disk I/O
processed_data = None
original_dataset = None

# Get data paths from environment
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH', 'processed_data.csv')
ORIGINAL_DATA_PATH = os.getenv('ORIGINAL_DATA_PATH', "BiztelAI_DS_Dataset_Mar'25.json")

# Load data function with caching
def load_processed_data():
    """Load processed dataset with caching for performance."""
    global processed_data
    if processed_data is None:
        try:
            processed_data = pd.read_csv(PROCESSED_DATA_PATH)
            app.logger.info(f"Loaded processed data from {PROCESSED_DATA_PATH}. Shape: {processed_data.shape}")
        except Exception as e:
            app.logger.error(f"Error loading processed data: {str(e)}")
            app.logger.error(traceback.format_exc())
            processed_data = None
    return processed_data

def load_original_dataset():
    """Load original dataset with caching for performance."""
    global original_dataset
    if original_dataset is None:
        try:
            with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
                original_dataset = json.load(f)
            app.logger.info(f"Loaded original dataset from {ORIGINAL_DATA_PATH}. Contains {len(original_dataset)} conversations")
        except Exception as e:
            app.logger.error(f"Error loading original dataset: {str(e)}")
            app.logger.error(traceback.format_exc())
            original_dataset = None
    return original_dataset

# API Health Check Endpoint
@app.route('/api/health', methods=['GET'])
@track_request_time
def health_check():
    """Health check endpoint to verify the API is running."""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": os.getenv('FLASK_ENV', 'production')
        }), 200
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# API Documentation Endpoint
@app.route('/api', methods=['GET'])
@track_request_time
def api_documentation():
    """API documentation endpoint."""
    return jsonify({
        "api_name": "BiztelAI Dataset API",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/api/health",
                "method": "GET",
                "description": "Health check endpoint"
            },
            {
                "path": "/api/dataset/summary",
                "method": "GET",
                "description": "Get processed dataset summary"
            },
            {
                "path": "/api/transform",
                "method": "POST",
                "description": "Transform raw data into processed form"
            },
            {
                "path": "/api/analyze/transcript",
                "method": "POST",
                "description": "Analyze a chat transcript and get insights"
            },
            {
                "path": "/api/metrics",
                "method": "GET",
                "description": "Get API performance metrics"
            }
        ]
    }), 200

# Endpoint 1: Fetch and return processed dataset summary
@app.route('/api/dataset/summary', methods=['GET'])
@track_request_time
def get_dataset_summary():
    """
    Return a summary of the processed dataset including statistics and key metrics.
    
    Optional query parameters:
    - refresh: If set to true, bypass the cache and generate a new summary
    - level: Summary level (basic, detailed) - default is basic
    """
    global dataset_summary_cache, cache_timestamp
    
    try:
        # Check if we should refresh the cache
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        summary_level = request.args.get('level', 'basic').lower()
        
        # Use cached summary if available and not expired
        current_time = time.time()
        if (not refresh and 
            dataset_summary_cache is not None and 
            cache_timestamp is not None and 
            current_time - cache_timestamp < CACHE_EXPIRY and
            summary_level in dataset_summary_cache):
            
            app.logger.info(f"Using cached dataset summary (level: {summary_level})")
            return jsonify(dataset_summary_cache[summary_level]), 200
        
        # Load the data
        df = load_processed_data()
        if df is None:
            return jsonify({"error": "Failed to load dataset"}), 500
        
        # Initialize cache if needed
        if dataset_summary_cache is None:
            dataset_summary_cache = {}
        
        # Generate basic summary
        basic_summary = {
            "dataset_shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "unique_conversations": df['conversation_id'].nunique(),
            "unique_articles": df['article_url'].nunique(),
            "agent_distribution": df['agent'].value_counts().to_dict(),
            "sentiment_distribution": df['sentiment'].value_counts().to_dict(),
            "config_distribution": df['config'].value_counts().to_dict(),
            "message_stats": {
                "avg_message_length": float(df['message_length'].mean()),
                "avg_word_count": float(df['word_count'].mean()),
                "min_message_length": int(df['message_length'].min()),
                "max_message_length": int(df['message_length'].max())
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Store in cache
        dataset_summary_cache['basic'] = basic_summary
        cache_timestamp = current_time
        
        # Generate detailed summary if requested
        if summary_level == 'detailed':
            # Article-level summary
            article_summary = df.groupby('article_url').agg(
                message_count=('message', 'count'),
                conversation_count=('conversation_id', 'nunique'),
                avg_message_length=('message_length', 'mean'),
                avg_word_count=('word_count', 'mean')
            ).sort_values('message_count', ascending=False)
            
            # Convert to dictionary for JSON serialization
            article_dict = article_summary.head(10).reset_index().to_dict(orient='records')
            
            # Agent-level summary
            agent_summary = df.groupby('agent').agg(
                message_count=('message', 'count'),
                avg_message_length=('message_length', 'mean'),
                avg_word_count=('word_count', 'mean')
            ).reset_index().to_dict(orient='records')
            
            # Sentiment by agent
            sentiment_by_agent = pd.crosstab(df['agent'], df['sentiment'])
            sentiment_by_agent_pct = sentiment_by_agent.div(sentiment_by_agent.sum(axis=1), axis=0) * 100
            
            # Convert to dictionary for JSON serialization
            sentiment_by_agent_dict = {
                agent: sentiment_by_agent_pct.loc[agent].to_dict() 
                for agent in sentiment_by_agent_pct.index
            }
            
            detailed_summary = {
                **basic_summary,
                "top_articles": article_dict,
                "agent_summary": agent_summary,
                "sentiment_by_agent": sentiment_by_agent_dict,
                "turn_rating_distribution": df['turn_rating'].value_counts().to_dict()
            }
            
            # Store in cache
            dataset_summary_cache['detailed'] = detailed_summary
            
            return jsonify(detailed_summary), 200
        
        return jsonify(basic_summary), 200
    
    except Exception as e:
        app.logger.error(f"Error generating dataset summary: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to generate dataset summary: {str(e)}"}), 500

# Endpoint 2: Perform real-time data transformation
@app.route('/api/transform', methods=['POST'])
@track_request_time
def transform_data():
    """
    Transform raw data into processed form using the same pipeline as the dataset.
    
    Expected input:
    - JSON data in the same format as the original dataset
    
    Returns:
    - Processed data with all transformations applied
    """
    try:
        # Get JSON data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Initialize data pipeline components
        data_loader = DataLoader("temp")  # Dummy path as we'll use data directly
        cleaner = DataCleaner(None)  # Will set dataframe later
        preprocessor = TextPreprocessor(None)  # Will set dataframe later
        transformer = FeatureTransformer(None)  # Will set dataframe later
        
        # Convert input data to dataframe
        conversations = []
        
        try:
            # Process each conversation in the input
            for conversation_id, conversation_data in data.items():
                article_url = conversation_data.get('article_url', '')
                config = conversation_data.get('config', '')
                
                for message in conversation_data.get('content', []):
                    message_data = {
                        'conversation_id': conversation_id,
                        'article_url': article_url,
                        'config': config,
                        'message': message.get('message', ''),
                        'agent': message.get('agent', ''),
                        'sentiment': message.get('sentiment', ''),
                        'knowledge_source': ','.join(message.get('knowledge_source', [])) if isinstance(message.get('knowledge_source'), list) else '',
                        'turn_rating': message.get('turn_rating', '')
                    }
                    conversations.append(message_data)
            
            df = pd.DataFrame(conversations)
            
            # Apply data cleaning
            cleaner.df = df
            df = cleaner.handle_missing_values()
            df = cleaner.remove_duplicates()
            df = cleaner.clean_text_data()
            df = cleaner.check_correct_data_types()
            
            # Apply text preprocessing
            preprocessor.df = df
            df = preprocessor.tokenize()
            df = preprocessor.remove_stopwords()
            df = preprocessor.lemmatize()
            
            # Apply feature transformation
            transformer.df = df
            cat_columns = ['agent', 'config', 'turn_rating', 'sentiment']
            df = transformer.encode_categorical(cat_columns)
            df = transformer.create_features()
            
            # Convert DataFrame to records for JSON serialization
            result = df.to_dict(orient='records')
            
            return jsonify({
                "transformed_data": result,
                "record_count": len(result),
                "transformations_applied": [
                    "missing_value_handling",
                    "duplicate_removal",
                    "text_cleaning",
                    "tokenization",
                    "stopword_removal",
                    "lemmatization",
                    "categorical_encoding",
                    "feature_creation"
                ]
            }), 200
            
        except Exception as e:
            app.logger.error(f"Error in data transformation pipeline: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({"error": f"Data transformation failed: {str(e)}"}), 500
    
    except Exception as e:
        app.logger.error(f"Error processing transformation request: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Endpoint 3: Allow users to send input and receive processed insights
@app.route('/api/analyze/transcript', methods=['POST'])
@track_request_time
def analyze_transcript_api():
    """
    Analyze a chat transcript and return insights.
    
    Expected input:
    - JSON data containing a chat transcript
    
    Returns:
    - Analysis results including article link, message counts, and sentiment analysis
    """
    try:
        # Get JSON data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        transcript_data = request.get_json()
        
        # Validate input
        if not transcript_data:
            return jsonify({"error": "No transcript data provided"}), 400
        
        # Analyze transcript
        results = analyze_chat_transcript(transcript_data)
        
        # Check for errors
        if 'error' in results:
            return jsonify({"error": results['error']}), 400
        
        # Return results
        return jsonify({
            "analysis_results": results,
            "analysis_timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        app.logger.error(f"Error in transcript analysis: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Performance metrics endpoint
@app.route('/api/metrics', methods=['GET'])
@track_request_time
def get_api_metrics():
    """
    Return API performance metrics.
    
    Returns:
    - Request counts, average response times, and error rates
    """
    try:
        if not request_metrics:
            return jsonify({"metrics": "No metrics collected yet"}), 200
        
        # Calculate metrics
        endpoints = {}
        total_requests = len(request_metrics)
        total_errors = 0
        total_duration = 0
        
        for req_id, data in request_metrics.items():
            endpoint = data['endpoint']
            duration = data['duration']
            status_code = data['status_code']
            
            # Initialize endpoint metrics if not exists
            if endpoint not in endpoints:
                endpoints[endpoint] = {
                    'request_count': 0,
                    'error_count': 0,
                    'total_duration': 0,
                    'avg_duration': 0
                }
            
            # Update endpoint metrics
            endpoints[endpoint]['request_count'] += 1
            endpoints[endpoint]['total_duration'] += duration
            
            if status_code >= 400:
                endpoints[endpoint]['error_count'] += 1
                total_errors += 1
            
            total_duration += duration
        
        # Calculate averages
        for endpoint, data in endpoints.items():
            data['avg_duration'] = data['total_duration'] / data['request_count']
        
        # Calculate overall metrics
        overall_metrics = {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': (total_errors / total_requests) if total_requests > 0 else 0,
            'avg_response_time': (total_duration / total_requests) if total_requests > 0 else 0
        }
        
        return jsonify({
            "overall_metrics": overall_metrics,
            "endpoint_metrics": endpoints,
            "generated_at": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        app.logger.error(f"Error generating API metrics: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to generate API metrics: {str(e)}"}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    app.logger.info(f"404 error: {request.path}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"500 error: {error}")
    app.logger.error(traceback.format_exc())
    return jsonify({"error": "Internal server error"}), 500

# Preload data when the server starts to improve initial response times
with app.app_context():
    app.logger.info("Preloading data...")
    load_processed_data()
    load_original_dataset()
    app.logger.info("Data preloading completed.")

if __name__ == '__main__':
    # Get server config from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    flask_env = os.getenv('FLASK_ENV', 'production')
    
    # Enable profiler in development mode if specified
    if flask_env == 'development' and os.getenv('PROFILER_ENABLED', 'false').lower() == 'true':
        app.config['PROFILE'] = True
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
        app.logger.info("Profiler enabled")
    
    app.logger.info(f"Starting BiztelAI Dataset API in {flask_env} mode...")
    app.run(host=host, port=port, threaded=True, debug=(flask_env == 'development')) 