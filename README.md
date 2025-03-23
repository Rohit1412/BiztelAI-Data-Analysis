# BiztelAI Dataset Analysis Project

This project implements a comprehensive analysis pipeline for the BiztelAI dataset, which contains chat transcripts between agents discussing Washington Post articles. The implementation is divided into four main tasks:

## Task 1: Data Ingestion and Preprocessing

A modular data pipeline for loading, cleaning, and preprocessing the BiztelAI dataset.

### Features

- **Data Loading**: JSON parsing with validation
- **Data Cleaning**: Handling missing values, duplicates, and data type consistency
- **Text Preprocessing**: Tokenization, lemmatization, and stopword removal
- **Feature Engineering**: Extracting insights from text data
- **OOP Architecture**: Modular components with clear responsibilities

### Key Files

- `data_pipeline.py`: Main pipeline implementation with OOP design
- `test_pipeline.py`: Validation script for the pipeline
- `visualize_data.py`: Generates visualizations of the processed data
- `analyze_stats.py`: Statistical analysis of the dataset
- `TASK1SUMMARY.md`: Detailed summary of Task 1 implementation

## Task 2: Exploratory Data Analysis and Transcript Analysis

Advanced exploratory data analysis of the processed dataset and a specialized transcript analyzer.

### Features

- **Comprehensive EDA**: Statistical analysis and visualizations of patterns
- **Conversation Analysis**: Insights about agent interactions and message content
- **Sentiment Analysis**: Analysis of emotional patterns using DistilBERT
- **Topic Inference**: Detection of article topics from conversation content
- **Visualization**: Interactive plots for data exploration

### Key Files

- `eda.py`: Main EDA implementation with visualization capabilities
- `transcript_analyzer.py`: Specialized analyzer for individual transcripts
- `api.py`: API interface for transcript analysis
- `EDA_REPORT.md`: Comprehensive EDA findings report
- `TASK2_SUMMARY.md`: Detailed summary of Task 2 implementation

## Task 3: REST API with Flask

A RESTful API exposing dataset insights, data transformation, and transcript analysis.

### Features

- **Dataset Summary Endpoint**: Statistics and insights from the processed dataset
- **Data Transformation Endpoint**: Real-time processing of raw input data
- **Transcript Analysis Endpoint**: Insights extraction from chat transcripts
- **Performance Optimization**: Caching, request tracking, and scalability
- **Comprehensive Error Handling**: Structured errors with detailed logging
- **Containerization**: Docker support for easy deployment

### Key Files

- `api_server.py`: Main API implementation
- `test_api.py`: API testing script
- `README_API.md`: API documentation
- `Dockerfile` & `docker-compose.yml`: Container configuration
- `TASK3_SUMMARY.md`: Detailed summary of Task 3 implementation

## Task 4: Object-Oriented Programming Enhancements

### Features
- **Abstract Base Class Architecture**: Implemented a flexible data processor hierarchy based on the abstract `DataProcessor` class
- **Design Pattern Implementation**: Utilized Decorator, Composite, Factory, and Template Method patterns
- **Configuration-Driven Pipeline**: Created a JSON-based configuration system for building data pipelines
- **Enhanced Observability**: Added comprehensive metadata tracking, timing, and logging throughout the pipeline
- **Unit Testing**: Developed tests to verify the OOP architecture and design patterns

### Key Files
- `data_processor.py`: Core OOP framework with abstract base class, decorators, and factory
- `data_pipeline.py`: Refactored concrete processor implementations
- `pipeline_config.json`: Configuration file for pipeline definition
- `run_pipeline.py`: Script to run the pipeline from configuration 
- `class_diagram.md`: Visual representation of the architecture
- `test_oop_pipeline.py`: Unit tests for the OOP implementation
- `TASK4_SUMMARY.md`: Detailed summary of the OOP enhancements

## Performance Testing

The project includes tools for performance testing the API under various load conditions to identify bottlenecks and ensure optimal performance in production environments.

### Features
- **Parallel Request Testing**: Test API endpoints with concurrent requests
- **Performance Metrics**: Measure response times, success rates, and throughput
- **Authentication Testing**: Test performance with JWT authentication
- **Visualization**: Generate charts and graphs of performance metrics
- **Result Storage**: Save test results for trend analysis

### Key Files
- `performance_test.py`: Core performance testing tool
- `test_all_endpoints.py`: Script to test all API endpoints
- `PERFORMANCE_TESTING.md`: Documentation for performance testing options and interpreting results.

### Running Performance Tests

Test a single endpoint:
```
python performance_test.py --endpoint /api/dataset/summary --requests 100 --concurrency 10
```

Test all endpoints:
```
python test_all_endpoints.py --requests 50 --concurrency 5 --visualize
```

See `PERFORMANCE_TESTING.md` for detailed documentation on performance testing options and interpreting results.

## Getting Started

### Prerequisites

- Python 3.9+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Running the Project

#### Data Pipeline (Task 1)

```
python data_pipeline.py
```

This will:
- Load the raw JSON dataset
- Clean and preprocess the data
- Save the processed data as CSV

#### Exploratory Data Analysis (Task 2)

```
python eda.py
```

This will:
- Generate statistical insights about the dataset
- Create visualizations in the `eda_visualizations` directory
- Produce a comprehensive analysis of patterns

#### Transcript Analysis (Task 2)

```
python transcript_analyzer.py
```

Example of analyzing a transcript:
```python
from transcript_analyzer import analyze_chat_transcript

results = analyze_chat_transcript(transcript_data)
```

#### API Server (Task 3)

```
python run_api_server.py
```

This will:
- Check dependencies and configuration
- Start the Flask API server
- Open API documentation in the browser

Alternatively, use Docker:
```
docker-compose up
```

### API Usage

See `README_API.md` for detailed API documentation, including:
- Endpoint descriptions
- Request/response formats
- Example usage in various programming languages
- Performance considerations

## Project Structure

```
├── data_pipeline.py         # Task 1: Data ingestion pipeline
├── data_processor.py        # Task 4: Abstract base class for data processors
├── eda.py                   # Task 2: Exploratory data analysis
├── transcript_analyzer.py   # Task 2: Transcript analysis tool
├── api_server.py            # Task 3: REST API implementation
├── test_api.py              # Task 3: API testing script
├── run_api_server.py        # Helper script to run the API server
├── performance_test.py      # Performance testing tool
├── test_all_endpoints.py    # Script to test all API endpoints
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── requirements.txt         # Project dependencies
├── README.md                # Main project documentation
├── README_API.md            # API documentation
├── PERFORMANCE_TESTING.md   # Performance testing documentation
├── PERFORMANCE_SUMMARY.md   # Summary of performance optimizations
├── TASK1SUMMARY.md          # Task 1 summary
├── TASK2_SUMMARY.md         # Task 2 summary
├── TASK3_SUMMARY.md         # Task 3 summary
├── TASK4_SUMMARY.md         # Task 4 summary
└── EDA_REPORT.md            # Exploratory data analysis report
```

## License

This project is proprietary and confidential.

## Acknowledgements

- Dataset provided by BiztelAI
- Based on Washington Post articles