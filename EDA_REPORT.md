# Exploratory Data Analysis (EDA) Report - BiztelAI Dataset

## Overview

This report presents the findings from an exploratory data analysis of the BiztelAI dataset, which contains chat transcripts between agents discussing articles from The Washington Post. The analysis focuses on understanding patterns in agent interactions, message content, sentiment distribution, and article-related patterns.

## Dataset Structure

The dataset consists of:
- **11,760 messages** across **539 unique conversations**
- Each conversation references a Washington Post article and involves two agents:
  - **agent_1** (51.6% of messages)
  - **agent_2** (48.4% of messages)
- Each message includes:
  - The text content
  - Agent identifier
  - Sentiment label
  - Knowledge source
  - Turn rating

## Conversation Patterns

### Message Distribution

- Average of **21.82 messages** per conversation
- Range from 20 to 33 messages per conversation
- Agent 1 averages **11.26 messages** per conversation
- Agent 2 averages **10.56 messages** per conversation

### Turn-Taking Patterns

Our analysis of turn-taking patterns shows:
- Conversations have a structured back-and-forth pattern between agents
- After agent_1 speaks, agent_2 responds 99.8% of the time
- After agent_2 speaks, agent_1 responds 99.9% of the time
- Very few instances where the same agent sends multiple consecutive messages

## Message Content Analysis

### Message Length

- Average message length: **102.58 characters**
- Average word count: **20.02 words**
- 25th percentile: 64 characters / 13 words
- 75th percentile: 131 characters / 25 words

### Common Vocabulary

Different word usage patterns were observed between agents:

#### Agent 1:
- Uses more questioning vocabulary and initiates topics
- Frequently uses words like "think," "know," "interesting," "article," and "question"
- More likely to ask questions (higher frequency of question marks)

#### Agent 2:
- Uses more responsive and explanatory vocabulary 
- Frequently uses words like "think," "know," "would," "yes," and "good"
- More likely to provide explanations and opinions

## Sentiment Analysis

### Overall Sentiment Distribution

The most common sentiment labels in the dataset:
1. **Curious to dive deeper** (47.13%)
2. **Neutral** (24.17%)
3. Other sentiments include: Happy, Surprised, Fearful, Sad, Angry, and Disgusted

### Sentiment by Agent

- **Agent 1** expresses more curiosity:
  - "Curious to dive deeper" sentiment: 64.25%
  - "Neutral" sentiment: 17.82%

- **Agent 2** has more neutral responses:
  - "Neutral" sentiment: 30.92% 
  - "Curious to dive deeper" sentiment: 29.11%
  - More varied emotional responses

### Sentiment Evolution

Analyzing sentiment throughout conversations:
- Initial messages (positions 1-3) show higher curiosity
- Middle messages (positions 4-15) show more varied sentiments
- Final messages (positions 16+) show increasing positivity and resolution
- This suggests a pattern of initial curiosity → exploration → resolution

## Article-Related Patterns

### Article Type Distribution

Based on URL analysis:
- **Sports articles**: 51.3%
- **Politics articles**: 23.7%
- **Other/General news**: 15.4%
- **Business/Economic**: 9.6%

### Article Impact on Conversation

- **Sports articles** generate more positive and curious sentiment
- **Politics articles** generate more neutral and occasionally negative sentiment
- **Business articles** generate more analytical and neutral sentiment

### Article Configuration Analysis

The dataset includes 4 configurations (A, B, C, D) with different distributions:
- **Config D**: 54.0%
- **Config C**: 16.4%
- **Config B**: 14.8%
- **Config A**: 14.8%

Different configurations show slightly different sentiment patterns, suggesting they may represent different conversation guidance or instructions.

## Turn Rating Analysis

Turn ratings assess the quality of each message:
- **"Good"** is the most common rating: 69.2%
- **"Excellent"** represents 15.8% of messages
- **"Passable"** represents 10.2% of messages
- Lower ratings are less frequent

Correlation with other features:
- Higher turn ratings correlate with longer, more substantive messages
- Higher turn ratings correlate with more emotionally expressive messages
- Lower turn ratings correlate with very short or generic responses

## LLM-Based Transcript Analysis

We implemented a lightweight LLM-based analysis tool using DistilBERT to:
1. Automatically determine the article topic from conversation content
2. Count messages by each agent
3. Analyze the sentiment distribution for each agent

### Model Performance

The sentiment analysis model achieved:
- **76.3% accuracy** when comparing to the original sentiment labels
- Best performance on clearly positive or negative sentiments
- Lower performance on nuanced sentiments like "Curious to dive deeper"

### API Implementation

We created a Flask API for analyzing transcripts that:
- Takes a JSON conversation transcript as input
- Returns article information, message counts, and sentiment analysis
- Includes visualization capabilities
- Exposes endpoints for both direct JSON input and file uploads

## Insights and Observations

1. **Structured Conversations**: Dialogues follow a clear structure with balanced participation.

2. **Sentiment Progression**: Conversations typically move from curiosity to resolution.

3. **Agent Roles**: Agent 1 tends to initiate topics and ask questions, while Agent 2 tends to respond and elaborate.

4. **Content Impact**: Article topics influence the sentiment and engagement in conversations.

5. **Turn Quality Factors**: Longer, more substantive, and emotionally responsive messages receive higher turn ratings.

6. **Language Patterns**: Distinct vocabulary and language styles between agents suggest different conversational roles.

## Conclusion

The BiztelAI dataset provides rich information about structured conversations between agents discussing news articles. The patterns in agent interactions, sentiment flow, and language usage suggest that these conversations may be used for training dialogue systems or evaluating conversation quality. The LLM-based analysis demonstrates that automated tools can effectively extract key information from these transcripts with reasonable accuracy. 