# BiztelAI Dataset Analysis

This project provides exploratory data analysis (EDA) and an API for analyzing chat transcripts from the BiztelAI dataset.

## Project Components

### 1. Exploratory Data Analysis (EDA)

The EDA component analyzes the entire dataset to uncover patterns and insights in agent conversations about Washington Post articles.

Key files:
- `eda.py`: Main EDA script that generates visualizations and statistics
- `EDA_REPORT.md`: Comprehensive report of findings from the analysis

### 2. Transcript Analysis API

The API component allows analysis of individual chat transcripts to extract key information like article topics, message counts, and agent sentiments.

Key files:
- `transcript_analyzer.py`: Core analysis logic using a lightweight LLM
- `api.py`: Flask API that exposes the analysis functionality

## Setup Instructions

### Prerequisites

- Python 3.8+ installed
- Virtual environment (recommended)

### Installation

1. Clone the repository
2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download NLTK resources:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Running the Code

### Exploratory Data Analysis

Run the complete EDA pipeline:
```
python eda.py
```

This will generate visualizations in the `eda_visualizations` directory and output statistics to the console.

### Transcript Analysis

Analyze a specific transcript:
```
python transcript_analyzer.py
```

This will run an example analysis on the first conversation in the dataset and save visualizations to the `transcript_analysis` directory.

### API Server

Start the API server:
```
python api.py
```

The API will be available at http://localhost:5000 with the following endpoints:

- `GET /health` - Health check
- `POST /analyze` - Analyze a transcript provided in the request body
- `POST /analyze_file` - Analyze a transcript from an uploaded file
- `GET /analyze_by_id?conversation_id=<ID>` - Analyze a specific conversation by ID

## API Usage Examples

### Analyze by Conversation ID

```bash
curl -X GET "http://localhost:5000/analyze_by_id?conversation_id=t_d004c097-424d-45d4-8f91-833d85c2da31"
```

### Analyze JSON Data

```bash
curl -X POST -H "Content-Type: application/json" -d @sample_transcript.json http://localhost:5000/analyze
```

### Analyze Uploaded File

```bash
curl -X POST -F "transcript=@sample_transcript.json" http://localhost:5000/analyze_file
```

## Results

The analysis provides:
- Possible article link or topic
- Number of messages sent by each agent
- Overall sentiment analysis for each agent
- Visualization of the results

## Model Accuracy

The sentiment analysis model achieves approximately 76.3% accuracy when compared to the original sentiment labels in the dataset. The model performs best on clearly positive or negative sentiments, with lower performance on nuanced sentiments. 