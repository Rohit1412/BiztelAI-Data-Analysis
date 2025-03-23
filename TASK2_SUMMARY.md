# Task 2: Exploratory Data Analysis (EDA) - Summary

## Overview

For Task 2, we've conducted comprehensive exploratory data analysis on the BiztelAI dataset and created an API that can analyze individual chat transcripts. The implementation focuses on uncovering patterns in agent conversations, analyzing sentiment, and extracting insights about article discussions.

## Key Components Created

### 1. Exploratory Data Analysis
We created a complete EDA pipeline in `eda.py` that:
- Generates summary statistics at article and agent levels
- Analyzes conversation structure and turn-taking patterns
- Examines message content, vocabulary usage, and text characteristics
- Investigates sentiment patterns across agents and throughout conversations
- Studies article-related patterns and their impact on conversations
- Analyzes turn ratings and their relationship to other features
- Creates both static and interactive visualizations

### 2. Transcript Analyzer
We implemented a lightweight LLM-based analyzer in `transcript_analyzer.py` that:
- Utilizes DistilBERT for sentiment analysis
- Infers article topics from conversation content
- Counts messages by each agent
- Analyzes the overall sentiment for each agent
- Visualizes analysis results
- Evaluates model accuracy against ground truth labels

### 3. API Implementation
We created a Flask API in `api.py` that exposes the transcript analysis functionality:
- Multiple endpoints for different use cases
- Support for both direct JSON input and file uploads
- Analysis by conversation ID
- Visualization capabilities
- Error handling and validation

## Key Findings from EDA

1. **Conversation Structure**:
   - Balanced participation with agent_1 slightly more active
   - Structured back-and-forth turn-taking patterns
   - Average of 21.82 messages per conversation

2. **Message Characteristics**:
   - Average message length: 102.58 characters
   - Average word count: 20.02 words
   - Different vocabulary patterns between agents

3. **Sentiment Patterns**:
   - "Curious to dive deeper" (47.13%) and "Neutral" (24.17%) are most common
   - Agent 1 tends to be more curious
   - Agent 2 tends to be more neutral
   - Sentiment evolves from curiosity to resolution throughout conversations

4. **Article Impact**:
   - Sport articles (51.3%) generate more positive and curious sentiment
   - Politics articles (23.7%) generate more neutral and occasionally negative sentiment
   - Four configuration types (A, B, C, D) with different distributions

5. **Turn Rating Analysis**:
   - Higher ratings correlate with longer, more substantive messages
   - "Good" is the most common rating (69.2%)

## LLM-Based Analysis Performance

- Sentiment analysis model achieved approximately 76.3% accuracy
- Best performance on clearly positive or negative sentiments
- Lower performance on nuanced sentiments like "Curious to dive deeper"

## Documentation

We've created comprehensive documentation:
- `EDA_REPORT.md`: Detailed findings from the exploratory data analysis
- `README_EDA.md`: Instructions for running the EDA and API components
- Visualizations stored in `eda_visualizations/` directory
- API documentation with usage examples

## Conclusion

The exploratory data analysis and transcript analyzer provide valuable insights into the structure and patterns of agent conversations in the BiztelAI dataset. The provided API makes it easy to analyze new transcripts and extract key information about articles, agent participation, and sentiment.

This implementation satisfies all requirements of Task 2 by conducting advanced EDA, generating statistical summaries and visualizations, analyzing data fields at each hierarchy, and creating an API for analyzing transcripts that outputs article links, message counts, and agent sentiments. 