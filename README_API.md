<<<<<<< HEAD
# BiztelAI Dataset REST API

This README documents the REST API implementation for Task 3 of the BiztelAI Dataset project.

## Overview

The API provides a comprehensive interface to the BiztelAI dataset, offering endpoints for data summary, real-time data transformation, and transcript analysis. The implementation uses Flask and follows REST principles, with JSON as the primary data format.

## Features

- **Dataset Summary**: Fetch statistical insights about the processed dataset
- **Real-time Data Transformation**: Transform raw input data using the same pipeline as the dataset
- **Transcript Analysis**: Analyze chat transcripts to extract insights
- **Performance Optimization**: Caching, request tracking, and performance metrics
- **Comprehensive Error Handling**: Structured error responses with logging
- **API Documentation**: Self-documenting API with endpoint descriptions

## API Endpoints

### Health Check
```
GET /api/health
```
Checks if the API is running correctly.

### API Documentation
```
GET /api
```
Returns information about all available endpoints.

### Dataset Summary
```
GET /api/dataset/summary
```
Returns statistics and insights from the processed dataset.

**Query Parameters:**
- `level`: Summary detail level (`basic` or `detailed`), default is `basic`
- `refresh`: Force regeneration of the summary instead of using cache (`true` or `false`), default is `false`

### Data Transformation
```
POST /api/transform
```
Transforms raw data using the complete data pipeline.

**Request Body:** JSON data in the same format as the original dataset

**Transformations Applied:**
- Missing value handling
- Duplicate removal
- Text cleaning
- Tokenization
- Stopword removal
- Lemmatization
- Categorical encoding
- Feature creation

### Transcript Analysis
```
POST /api/analyze/transcript
```
Analyzes a chat transcript and returns insights.

**Request Body:** JSON data containing a chat transcript

**Response:** Analysis including article links, message counts by agent, and sentiment analysis

### API Metrics
```
GET /api/metrics
```
Returns performance metrics about the API, including request counts, response times, and error rates.

## Performance Optimization

The API includes several optimizations for performance and scalability:

1. **Caching**: The dataset summary is cached with configurable expiry to reduce computation
2. **Data Preloading**: Datasets are loaded on server startup to minimize initial request latency
3. **Request Tracking**: All requests are tracked for performance analysis
4. **Profiler Integration**: Development mode includes a profiler for identifying bottlenecks
5. **Efficient Error Handling**: Structured exception handling with minimal overhead
6. **Threaded Server**: The server runs in threaded mode to handle concurrent requests

## Error Handling

The API implements a comprehensive error handling system:

- **Structured Error Responses**: All errors return JSON with error messages
- **Status Codes**: Appropriate HTTP status codes (400, 404, 500) for different error types
- **Logging**: Detailed logging for all errors with stacktraces
- **Request Tracking**: Each request has a unique ID for tracing in logs

## Logging

The API uses a rotating file handler for logging:

- Log files are stored in the `logs` directory
- Maximum file size is 10MB with 5 backup files
- Both file and console logging is configured
- All requests and errors are logged with timestamps

## Testing

A comprehensive test suite is included in `test_api.py` that validates all API endpoints:

```
python test_api.py
```

This will test all endpoints with sample data and report on their functionality.

## Requirements

The API requires the following Python packages:
- Flask
- pandas
- numpy
- Werkzeug
- colorama (for test script)
- requests (for test script)

These dependencies are included in the project's `requirements.txt` file.

## Usage Examples

### Fetch Dataset Summary

```python
import requests

# Get basic summary
response = requests.get("http://localhost:5000/api/dataset/summary")
basic_summary = response.json()

# Get detailed summary
response = requests.get("http://localhost:5000/api/dataset/summary?level=detailed")
detailed_summary = response.json()
```

### Transform Raw Data

```python
import requests
import json

# Sample conversation data
conversation_data = {
    "conversation_123": {
        "article_url": "https://www.washingtonpost.com/example",
        "config": "A",
        "content": [
            {
                "message": "What did you think of the article?",
                "agent": "agent_1",
                "sentiment": "Curious to dive deeper",
                "knowledge_source": ["Article"],
                "turn_rating": "Good"
            },
            {
                "message": "I found it very informative.",
                "agent": "agent_2",
                "sentiment": "Happy",
                "knowledge_source": ["Article"],
                "turn_rating": "Good"
            }
        ]
    }
}

# Send data for transformation
response = requests.post(
    "http://localhost:5000/api/transform",
    json=conversation_data,
    headers={"Content-Type": "application/json"}
)

# Get transformed data
transformed_data = response.json()
```

### Analyze Transcript

```python
import requests
import json

# Sample transcript data
transcript_data = {
    "conversation_123": {
        "article_url": "https://www.washingtonpost.com/example",
        "config": "A",
        "content": [
            {
                "message": "What did you think of the article?",
                "agent": "agent_1",
                "sentiment": "Curious to dive deeper",
                "knowledge_source": ["Article"],
                "turn_rating": "Good"
            },
            {
                "message": "I found it very informative.",
                "agent": "agent_2",
                "sentiment": "Happy",
                "knowledge_source": ["Article"],
                "turn_rating": "Good"
            }
        ]
    }
}

# Send transcript for analysis
response = requests.post(
    "http://localhost:5000/api/analyze/transcript",
    json=transcript_data,
    headers={"Content-Type": "application/json"}
)

# Get analysis results
analysis_results = response.json()
```

## Running the API

To start the API server:

```
python api_server.py
```

By default, the server will run on `http://localhost:5000`. Set the environment variable `PORT` to change the port number.

For development mode with profiling:

```
export FLASK_ENV=development
python api_server.py
=======
# BiztelAI Dataset REST API

This README documents the REST API implementation for Task 3 of the BiztelAI Dataset project.

## Overview

The API provides a comprehensive interface to the BiztelAI dataset, offering endpoints for data summary, real-time data transformation, and transcript analysis. The implementation uses Flask and follows REST principles, with JSON as the primary data format.

## Features

- **Dataset Summary**: Fetch statistical insights about the processed dataset
- **Real-time Data Transformation**: Transform raw input data using the same pipeline as the dataset
- **Transcript Analysis**: Analyze chat transcripts to extract insights
- **Performance Optimization**: Caching, request tracking, and performance metrics
- **Comprehensive Error Handling**: Structured error responses with logging
- **API Documentation**: Self-documenting API with endpoint descriptions

## API Endpoints

### Health Check
```
GET /api/health
```
Checks if the API is running correctly.

### API Documentation
```
GET /api
```
Returns information about all available endpoints.

### Dataset Summary
```
GET /api/dataset/summary
```
Returns statistics and insights from the processed dataset.

**Query Parameters:**
- `level`: Summary detail level (`basic` or `detailed`), default is `basic`
- `refresh`: Force regeneration of the summary instead of using cache (`true` or `false`), default is `false`

### Data Transformation
```
POST /api/transform
```
Transforms raw data using the complete data pipeline.

**Request Body:** JSON data in the same format as the original dataset

**Transformations Applied:**
- Missing value handling
- Duplicate removal
- Text cleaning
- Tokenization
- Stopword removal
- Lemmatization
- Categorical encoding
- Feature creation

### Transcript Analysis
```
POST /api/analyze/transcript
```
Analyzes a chat transcript and returns insights.

**Request Body:** JSON data containing a chat transcript

**Response:** Analysis including article links, message counts by agent, and sentiment analysis

### API Metrics
```
GET /api/metrics
```
Returns performance metrics about the API, including request counts, response times, and error rates.

## Performance Optimization

The API includes several optimizations for performance and scalability:

1. **Caching**: The dataset summary is cached with configurable expiry to reduce computation
2. **Data Preloading**: Datasets are loaded on server startup to minimize initial request latency
3. **Request Tracking**: All requests are tracked for performance analysis
4. **Profiler Integration**: Development mode includes a profiler for identifying bottlenecks
5. **Efficient Error Handling**: Structured exception handling with minimal overhead
6. **Threaded Server**: The server runs in threaded mode to handle concurrent requests

## Error Handling

The API implements a comprehensive error handling system:

- **Structured Error Responses**: All errors return JSON with error messages
- **Status Codes**: Appropriate HTTP status codes (400, 404, 500) for different error types
- **Logging**: Detailed logging for all errors with stacktraces
- **Request Tracking**: Each request has a unique ID for tracing in logs

## Logging

The API uses a rotating file handler for logging:

- Log files are stored in the `logs` directory
- Maximum file size is 10MB with 5 backup files
- Both file and console logging is configured
- All requests and errors are logged with timestamps

## Testing

A comprehensive test suite is included in `test_api.py` that validates all API endpoints:

```
python test_api.py
```

This will test all endpoints with sample data and report on their functionality.

## Requirements

The API requires the following Python packages:
- Flask
- pandas
- numpy
- Werkzeug
- colorama (for test script)
- requests (for test script)

These dependencies are included in the project's `requirements.txt` file.

## Usage Examples

### Fetch Dataset Summary

```python
import requests

# Get basic summary
response = requests.get("http://localhost:5000/api/dataset/summary")
basic_summary = response.json()

# Get detailed summary
response = requests.get("http://localhost:5000/api/dataset/summary?level=detailed")
detailed_summary = response.json()
```

### Transform Raw Data

```python
import requests
import json

# Sample conversation data
conversation_data = {
    "conversation_123": {
        "article_url": "https://www.washingtonpost.com/example",
        "config": "A",
        "content": [
            {
                "message": "What did you think of the article?",
                "agent": "agent_1",
                "sentiment": "Curious to dive deeper",
                "knowledge_source": ["Article"],
                "turn_rating": "Good"
            },
            {
                "message": "I found it very informative.",
                "agent": "agent_2",
                "sentiment": "Happy",
                "knowledge_source": ["Article"],
                "turn_rating": "Good"
            }
        ]
    }
}

# Send data for transformation
response = requests.post(
    "http://localhost:5000/api/transform",
    json=conversation_data,
    headers={"Content-Type": "application/json"}
)

# Get transformed data
transformed_data = response.json()
```

### Analyze Transcript

```python
import requests
import json

# Sample transcript data
transcript_data = {
    "conversation_123": {
        "article_url": "https://www.washingtonpost.com/example",
        "config": "A",
        "content": [
            {
                "message": "What did you think of the article?",
                "agent": "agent_1",
                "sentiment": "Curious to dive deeper",
                "knowledge_source": ["Article"],
                "turn_rating": "Good"
            },
            {
                "message": "I found it very informative.",
                "agent": "agent_2",
                "sentiment": "Happy",
                "knowledge_source": ["Article"],
                "turn_rating": "Good"
            }
        ]
    }
}

# Send transcript for analysis
response = requests.post(
    "http://localhost:5000/api/analyze/transcript",
    json=transcript_data,
    headers={"Content-Type": "application/json"}
)

# Get analysis results
analysis_results = response.json()
```

## Running the API

To start the API server:

```
python api_server.py
```

By default, the server will run on `http://localhost:5000`. Set the environment variable `PORT` to change the port number.

For development mode with profiling:

```
export FLASK_ENV=development
python api_server.py
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
``` 