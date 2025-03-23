<<<<<<< HEAD
# BiztelAI Dataset - Data Ingestion and Preprocessing 

## Overview

We've successfully implemented a modular data pipeline following object-oriented programming (OOP) principles to ingest, preprocess, and analyze the BiztelAI dataset of chat transcripts between agents discussing Washington Post articles.

## Data Pipeline Structure

The pipeline is structured using several independent but interconnected components:

1. **DataLoader**: Responsible for loading the JSON dataset and converting it into a pandas DataFrame.
2. **DataCleaner**: Handles missing values, duplicate records, data type conversion, and text cleaning.
3. **TextPreprocessor**: Performs NLP operations including tokenization, stopword removal, and lemmatization.
4. **FeatureTransformer**: Converts categorical variables to numerical representations and creates additional features.
5. **DataPipeline**: Orchestrates the entire process flow.

## Dataset Characteristics

After processing, we found:

- The dataset contains **11,760 messages** across **539 unique conversations**.
- Each conversation contains an average of **21.82 messages**.
- There are **two agents** (agent_1 and agent_2) with a balanced distribution (51.6% vs 48.4%).
- Four configuration types (A, B, C, D) with D being the most common (54.02%).
- Messages have an average length of **102.58 characters** and **20.02 words**.

## Preprocessing Steps Implemented

1. **Data Loading**:
   - Parsed nested JSON structure into a tabular format
   - Extracted conversation metadata (article URL, config) and message details

2. **Data Cleaning**:
   - Handled missing values (found and addressed)
   - Checked for and removed duplicates
   - Converted categorical columns to category data type for efficiency

3. **Text Preprocessing**:
   - Cleaned text by removing special characters and URLs
   - Tokenized messages
   - Removed stopwords
   - Applied lemmatization

4. **Feature Engineering**:
   - Encoded categorical variables (agent, config, turn_rating, sentiment)
   - Created additional features (message_length, word_count)

## Key Findings

1. **Message Characteristics**:
   - Message length varies considerably (from 2 to 463 characters)
   - Most messages contain between 13-25 words

2. **Sentiment Analysis**:
   - The most common sentiment is "Curious to dive deeper" (47.13%)
   - The second most common is "Neutral" (24.17%)

3. **Correlations**:
   - Strong correlation between message length and word count (0.984)
   - Weak positive correlation between sentiment and message length (0.105)
   - Weak positive correlation between agent and word count (0.070)

4. **Agent Behavior**:
   - Agent 1 tends to send slightly more messages per conversation than Agent 2
   - Agent 1 averages 11.26 messages per conversation
   - Agent 2 averages 10.56 messages per conversation

## Files Created

1. **data_pipeline.py**: Main pipeline implementation with OOP structure
2. **test_pipeline.py**: Test script to verify pipeline functionality
3. **visualize_data.py**: Script to generate data visualizations
4. **analyze_stats.py**: Statistical analysis of the processed data
5. **processed_data.csv**: The final processed dataset
6. Various visualization outputs (PNG files)

## Next Steps

Potential extensions for this project could include:

1. Topic modeling to identify common conversation themes
2. Sentiment analysis to understand emotional patterns in conversations
3. Building a classification model to predict conversation quality (based on turn_rating)
=======
# BiztelAI Dataset - Data Ingestion and Preprocessing 

## Overview

We've successfully implemented a modular data pipeline following object-oriented programming (OOP) principles to ingest, preprocess, and analyze the BiztelAI dataset of chat transcripts between agents discussing Washington Post articles.

## Data Pipeline Structure

The pipeline is structured using several independent but interconnected components:

1. **DataLoader**: Responsible for loading the JSON dataset and converting it into a pandas DataFrame.
2. **DataCleaner**: Handles missing values, duplicate records, data type conversion, and text cleaning.
3. **TextPreprocessor**: Performs NLP operations including tokenization, stopword removal, and lemmatization.
4. **FeatureTransformer**: Converts categorical variables to numerical representations and creates additional features.
5. **DataPipeline**: Orchestrates the entire process flow.

## Dataset Characteristics

After processing, we found:

- The dataset contains **11,760 messages** across **539 unique conversations**.
- Each conversation contains an average of **21.82 messages**.
- There are **two agents** (agent_1 and agent_2) with a balanced distribution (51.6% vs 48.4%).
- Four configuration types (A, B, C, D) with D being the most common (54.02%).
- Messages have an average length of **102.58 characters** and **20.02 words**.

## Preprocessing Steps Implemented

1. **Data Loading**:
   - Parsed nested JSON structure into a tabular format
   - Extracted conversation metadata (article URL, config) and message details

2. **Data Cleaning**:
   - Handled missing values (found and addressed)
   - Checked for and removed duplicates
   - Converted categorical columns to category data type for efficiency

3. **Text Preprocessing**:
   - Cleaned text by removing special characters and URLs
   - Tokenized messages
   - Removed stopwords
   - Applied lemmatization

4. **Feature Engineering**:
   - Encoded categorical variables (agent, config, turn_rating, sentiment)
   - Created additional features (message_length, word_count)

## Key Findings

1. **Message Characteristics**:
   - Message length varies considerably (from 2 to 463 characters)
   - Most messages contain between 13-25 words

2. **Sentiment Analysis**:
   - The most common sentiment is "Curious to dive deeper" (47.13%)
   - The second most common is "Neutral" (24.17%)

3. **Correlations**:
   - Strong correlation between message length and word count (0.984)
   - Weak positive correlation between sentiment and message length (0.105)
   - Weak positive correlation between agent and word count (0.070)

4. **Agent Behavior**:
   - Agent 1 tends to send slightly more messages per conversation than Agent 2
   - Agent 1 averages 11.26 messages per conversation
   - Agent 2 averages 10.56 messages per conversation

## Files Created

1. **data_pipeline.py**: Main pipeline implementation with OOP structure
2. **test_pipeline.py**: Test script to verify pipeline functionality
3. **visualize_data.py**: Script to generate data visualizations
4. **analyze_stats.py**: Statistical analysis of the processed data
5. **processed_data.csv**: The final processed dataset
6. Various visualization outputs (PNG files)

## Next Steps

Potential extensions for this project could include:

1. Topic modeling to identify common conversation themes
2. Sentiment analysis to understand emotional patterns in conversations
3. Building a classification model to predict conversation quality (based on turn_rating)
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
4. Time series analysis if temporal data becomes available 