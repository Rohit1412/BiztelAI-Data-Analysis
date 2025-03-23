<<<<<<< HEAD
import json
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TranscriptAnalyzer:
    """Class for analyzing a specific chat transcript using a light LLM."""
    
    def __init__(self):
        """Initialize the transcript analyzer with required models."""
        # Set up sentiment analysis pipeline using a lightweight model
        self.sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=self.sentiment_model, 
            tokenizer=self.sentiment_tokenizer,
            max_length=512,
            truncation=True
        )
        
        # Create output directory for results
        if not os.path.exists('transcript_analysis'):
            os.makedirs('transcript_analysis')
    
    def load_transcript(self, transcript_data):
        """Load a transcript from JSON data or file path."""
        try:
            if isinstance(transcript_data, str):
                # Check if it's a file path
                if os.path.isfile(transcript_data):
                    with open(transcript_data, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    # Try parsing as a JSON string
                    data = json.loads(transcript_data)
            else:
                # Assume it's already parsed JSON data
                data = transcript_data
            
            # Extract conversation ID (the key)
            if isinstance(data, dict) and len(data) == 1:
                conversation_id = list(data.keys())[0]
                conversation_data = data[conversation_id]
            else:
                # Handle case when single conversation is provided without ID
                conversation_id = "unknown_id"
                conversation_data = data
            
            # Convert to structured format
            messages = []
            
            if 'content' in conversation_data:
                article_url = conversation_data.get('article_url', '')
                config = conversation_data.get('config', '')
                
                for message in conversation_data['content']:
                    message_data = {
                        'conversation_id': conversation_id,
                        'article_url': article_url,
                        'config': config,
                        'message': message.get('message', ''),
                        'agent': message.get('agent', ''),
                        'sentiment': message.get('sentiment', ''),
                        'knowledge_source': ','.join(message.get('knowledge_source', [])) if isinstance(message.get('knowledge_source'), list) else message.get('knowledge_source', ''),
                        'turn_rating': message.get('turn_rating', '')
                    }
                    messages.append(message_data)
            
            transcript_df = pd.DataFrame(messages)
            return transcript_df
        
        except Exception as e:
            print(f"Error loading transcript: {e}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, texts):
        """Analyze sentiment of texts using the sentiment pipeline."""
        try:
            if not texts:
                return []
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Remove empty texts
            valid_texts = [text for text in processed_texts if text.strip()]
            
            if not valid_texts:
                return []
            
            # Get sentiment predictions
            results = self.sentiment_pipeline(valid_texts)
            
            # Convert binary sentiment labels to more nuanced labels
            mapped_results = []
            for result in results:
                label = result['label']
                score = result['score']
                
                if label == 'POSITIVE':
                    if score > 0.95:
                        mapped_label = 'Very Positive'
                    else:
                        mapped_label = 'Positive'
                else:  # NEGATIVE
                    if score > 0.95:
                        mapped_label = 'Very Negative'
                    else:
                        mapped_label = 'Negative'
                
                mapped_results.append({'label': mapped_label, 'score': score})
            
            return mapped_results
        
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return []
    
    def infer_article_topic(self, transcript_df):
        """Infer the article topic from the conversation content."""
        if transcript_df is None or transcript_df.empty:
            return "Unknown"
        
        # Get actual article URL if available
        article_url = transcript_df['article_url'].iloc[0] if not transcript_df['article_url'].iloc[0] == '' else None
        
        if article_url:
            return article_url
        
        # If no URL available, try to infer from content
        all_messages = ' '.join(transcript_df['message'].dropna().tolist())
        
        # Extract potential topics using keyword analysis
        sports_keywords = ['game', 'team', 'player', 'coach', 'score', 'season', 'match', 'championship', 'football', 'basketball', 'baseball', 'sports']
        politics_keywords = ['government', 'president', 'election', 'political', 'vote', 'congress', 'democrat', 'republican', 'policy', 'senate', 'law']
        business_keywords = ['company', 'market', 'stock', 'economy', 'business', 'finance', 'investment', 'trade', 'economic', 'industry', 'corporate']
        tech_keywords = ['technology', 'software', 'computer', 'app', 'internet', 'digital', 'tech', 'data', 'innovation', 'device', 'smartphone']
        
        # Count keyword occurrences
        topic_counts = {
            'Sports': sum(1 for keyword in sports_keywords if re.search(r'\b' + keyword + r'\b', all_messages, re.IGNORECASE)),
            'Politics': sum(1 for keyword in politics_keywords if re.search(r'\b' + keyword + r'\b', all_messages, re.IGNORECASE)),
            'Business': sum(1 for keyword in business_keywords if re.search(r'\b' + keyword + r'\b', all_messages, re.IGNORECASE)),
            'Technology': sum(1 for keyword in tech_keywords if re.search(r'\b' + keyword + r'\b', all_messages, re.IGNORECASE))
        }
        
        # Get topic with highest keyword count
        top_topic = max(topic_counts.items(), key=lambda x: x[1])
        
        if top_topic[1] > 0:
            return f"Likely a {top_topic[0]} article"
        else:
            return "Unknown article topic"
    
    def get_agent_message_counts(self, transcript_df):
        """Count messages by each agent."""
        if transcript_df is None or transcript_df.empty:
            return {}
        
        agent_counts = transcript_df['agent'].value_counts().to_dict()
        return agent_counts
    
    def analyze_agent_sentiment(self, transcript_df):
        """Analyze the overall sentiment for each agent."""
        if transcript_df is None or transcript_df.empty:
            return {}
        
        agent_sentiment = {}
        
        for agent in transcript_df['agent'].unique():
            agent_messages = transcript_df[transcript_df['agent'] == agent]['message'].tolist()
            
            # Use sentiment model to analyze
            sentiment_results = self.analyze_sentiment(agent_messages)
            
            if sentiment_results:
                # Count sentiment labels
                sentiment_counts = Counter([result['label'] for result in sentiment_results])
                
                # Calculate average sentiment score
                avg_score = sum([result['score'] for result in sentiment_results]) / len(sentiment_results)
                
                # Determine dominant sentiment
                dominant_sentiment = sentiment_counts.most_common(1)[0][0]
                
                # Calculate sentiment distribution percentages
                total = sum(sentiment_counts.values())
                sentiment_distribution = {sentiment: count / total * 100 for sentiment, count in sentiment_counts.items()}
                
                agent_sentiment[agent] = {
                    'dominant_sentiment': dominant_sentiment,
                    'avg_score': avg_score,
                    'sentiment_counts': dict(sentiment_counts),
                    'sentiment_distribution': sentiment_distribution
                }
            else:
                agent_sentiment[agent] = {'dominant_sentiment': 'Unknown', 'avg_score': 0.0}
        
        return agent_sentiment
    
    def analyze_transcript(self, transcript_data):
        """Analyze a transcript and return key insights."""
        # Load transcript
        transcript_df = self.load_transcript(transcript_data)
        
        if transcript_df is None or transcript_df.empty:
            return {
                'error': 'Failed to load or parse transcript data'
            }
        
        # Get article link/topic
        article_link = self.infer_article_topic(transcript_df)
        
        # Get message counts by agent
        message_counts = self.get_agent_message_counts(transcript_df)
        
        # Analyze sentiment by agent
        agent_sentiment = self.analyze_agent_sentiment(transcript_df)
        
        # Prepare results
        results = {
            'article_link': article_link,
            'message_counts': message_counts,
            'agent_sentiment': agent_sentiment,
            'conversation_id': transcript_df['conversation_id'].iloc[0] if not transcript_df.empty else 'unknown'
        }
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """Visualize analysis results."""
        if not results or 'error' in results:
            print("No valid results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Message counts
        agent_labels = list(results['message_counts'].keys())
        message_values = list(results['message_counts'].values())
        
        axes[0].bar(agent_labels, message_values, color=['#3498db', '#e74c3c'])
        axes[0].set_title('Message Count by Agent')
        axes[0].set_xlabel('Agent')
        axes[0].set_ylabel('Number of Messages')
        
        # Add count labels
        for i, count in enumerate(message_values):
            axes[0].text(i, count + 0.5, str(count), ha='center')
        
        # Sentiment analysis
        agent_sentiment = results['agent_sentiment']
        
        # Extract data for plotting
        agents = []
        sentiment_labels = []
        sentiment_values = []
        
        for agent, sentiment_data in agent_sentiment.items():
            for sentiment, percentage in sentiment_data.get('sentiment_distribution', {}).items():
                agents.append(agent)
                sentiment_labels.append(sentiment)
                sentiment_values.append(percentage)
        
        # Create DataFrame for plotting
        sentiment_df = pd.DataFrame({
            'Agent': agents,
            'Sentiment': sentiment_labels,
            'Percentage': sentiment_values
        })
        
        # Plot sentiment distribution as stacked bars
        if not sentiment_df.empty:
            sentiment_pivot = sentiment_df.pivot(index='Agent', columns='Sentiment', values='Percentage')
            sentiment_pivot.fillna(0, inplace=True)
            
            sentiment_pivot.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis')
            axes[1].set_title('Sentiment Distribution by Agent')
            axes[1].set_xlabel('Agent')
            axes[1].set_ylabel('Percentage (%)')
            axes[1].legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add article info
        fig.suptitle(f"Analysis of Conversation: {results['conversation_id']}\nArticle: {results['article_link']}", 
                     fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Results visualization saved to {save_path}")
        
        plt.close()
    
    def evaluate_model_accuracy(self, df, true_sentiment_col='sentiment', message_col='message', sample_size=100):
        """Evaluate the accuracy of the sentiment model against ground truth labels."""
        # Sample data for evaluation
        if len(df) > sample_size:
            eval_df = df.sample(sample_size, random_state=42)
        else:
            eval_df = df
        
        # Get predictions from model
        messages = eval_df[message_col].tolist()
        sentiment_results = self.analyze_sentiment(messages)
        
        if not sentiment_results:
            return {"error": "No sentiment predictions generated"}
        
        # Map ground truth labels to model's label space for comparison
        # This mapping depends on your dataset's specific sentiment labels
        sentiment_mapping = {
            'Neutral': 'Negative',  # Adjust based on your specific labels
            'Happy': 'Positive',
            'Positive': 'Positive',
            'Curious to dive deeper': 'Positive',
            'Surprised': 'Positive',
            'Sad': 'Negative',
            'Fearful': 'Negative',
            'Angry': 'Negative',
            'Disgusted': 'Negative',
            'Negative': 'Negative'
        }
        
        # Transform ground truth labels using mapping
        true_labels = eval_df[true_sentiment_col].map(lambda x: sentiment_mapping.get(x, 'Negative')).tolist()
        
        # Get predicted labels (simple positive/negative)
        pred_labels = [result['label'].split()[0] for result in sentiment_results]
        
        # Calculate accuracy and generate report
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels)
        
        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "sample_size": len(eval_df)
        }
        
        return results
    
    def run_accuracy_evaluation(self, data_path='processed_data.csv'):
        """Run accuracy evaluation on processed dataset."""
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            print(f"Loaded dataset for evaluation. Shape: {df.shape}")
            
            # Evaluate model accuracy
            eval_results = self.evaluate_model_accuracy(df)
            
            # Save results
            with open('transcript_analysis/model_evaluation.txt', 'w') as f:
                f.write("=== Sentiment Model Evaluation ===\n\n")
                f.write(f"Model: {self.sentiment_model_name}\n")
                f.write(f"Sample size: {eval_results.get('sample_size', 'N/A')}\n\n")
                f.write(f"Accuracy: {eval_results.get('accuracy', 'N/A'):.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(eval_results.get('classification_report', 'N/A'))
            
            print(f"Model evaluation saved to 'transcript_analysis/model_evaluation.txt'")
            return eval_results
            
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return {"error": str(e)}


# API function for transcript analysis
def analyze_chat_transcript(transcript_data):
    """
    Analyze a chat transcript and return key insights.
    
    Args:
        transcript_data: JSON data or file path to a transcript
    
    Returns:
        dict: Analysis results with the following information:
            - article_link: Possible article link or topic
            - message_counts: Number of messages by each agent
            - agent_sentiment: Overall sentiment analysis for each agent
    """
    analyzer = TranscriptAnalyzer()
    results = analyzer.analyze_transcript(transcript_data)
    
    # Save visualization
    if 'error' not in results:
        save_path = f"transcript_analysis/conversation_{results['conversation_id']}.png"
        analyzer.visualize_results(results, save_path)
    
    return results


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TranscriptAnalyzer()
    
    # Test with sample data or file
    try:
        # Load original dataset to extract a sample conversation
        with open("BiztelAI_DS_Dataset_Mar'25.json", 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Take first conversation as sample
        sample_convo_id = list(all_data.keys())[0]
        sample_convo = {sample_convo_id: all_data[sample_convo_id]}
        
        print(f"Analyzing sample conversation: {sample_convo_id}")
        
        # Analyze the sample conversation
        results = analyze_chat_transcript(sample_convo)
        
        # Print results
        print("\n=== Analysis Results ===")
        print(f"Article: {results['article_link']}")
        print("\nMessage Counts:")
        for agent, count in results['message_counts'].items():
            print(f"  {agent}: {count} messages")
        
        print("\nAgent Sentiment:")
        for agent, sentiment in results['agent_sentiment'].items():
            print(f"  {agent}: {sentiment['dominant_sentiment']} (score: {sentiment['avg_score']:.2f})")
            if 'sentiment_distribution' in sentiment:
                print("    Distribution:")
                for label, pct in sentiment['sentiment_distribution'].items():
                    print(f"      {label}: {pct:.1f}%")
        
        # Evaluate model accuracy
        print("\nEvaluating sentiment model accuracy...")
        analyzer.run_accuracy_evaluation()
        
    except Exception as e:
=======
import json
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TranscriptAnalyzer:
    """Class for analyzing a specific chat transcript using a light LLM."""
    
    def __init__(self):
        """Initialize the transcript analyzer with required models."""
        # Set up sentiment analysis pipeline using a lightweight model
        self.sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=self.sentiment_model, 
            tokenizer=self.sentiment_tokenizer,
            max_length=512,
            truncation=True
        )
        
        # Create output directory for results
        if not os.path.exists('transcript_analysis'):
            os.makedirs('transcript_analysis')
    
    def load_transcript(self, transcript_data):
        """Load a transcript from JSON data or file path."""
        try:
            if isinstance(transcript_data, str):
                # Check if it's a file path
                if os.path.isfile(transcript_data):
                    with open(transcript_data, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    # Try parsing as a JSON string
                    data = json.loads(transcript_data)
            else:
                # Assume it's already parsed JSON data
                data = transcript_data
            
            # Extract conversation ID (the key)
            if isinstance(data, dict) and len(data) == 1:
                conversation_id = list(data.keys())[0]
                conversation_data = data[conversation_id]
            else:
                # Handle case when single conversation is provided without ID
                conversation_id = "unknown_id"
                conversation_data = data
            
            # Convert to structured format
            messages = []
            
            if 'content' in conversation_data:
                article_url = conversation_data.get('article_url', '')
                config = conversation_data.get('config', '')
                
                for message in conversation_data['content']:
                    message_data = {
                        'conversation_id': conversation_id,
                        'article_url': article_url,
                        'config': config,
                        'message': message.get('message', ''),
                        'agent': message.get('agent', ''),
                        'sentiment': message.get('sentiment', ''),
                        'knowledge_source': ','.join(message.get('knowledge_source', [])) if isinstance(message.get('knowledge_source'), list) else message.get('knowledge_source', ''),
                        'turn_rating': message.get('turn_rating', '')
                    }
                    messages.append(message_data)
            
            transcript_df = pd.DataFrame(messages)
            return transcript_df
        
        except Exception as e:
            print(f"Error loading transcript: {e}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, texts):
        """Analyze sentiment of texts using the sentiment pipeline."""
        try:
            if not texts:
                return []
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Remove empty texts
            valid_texts = [text for text in processed_texts if text.strip()]
            
            if not valid_texts:
                return []
            
            # Get sentiment predictions
            results = self.sentiment_pipeline(valid_texts)
            
            # Convert binary sentiment labels to more nuanced labels
            mapped_results = []
            for result in results:
                label = result['label']
                score = result['score']
                
                if label == 'POSITIVE':
                    if score > 0.95:
                        mapped_label = 'Very Positive'
                    else:
                        mapped_label = 'Positive'
                else:  # NEGATIVE
                    if score > 0.95:
                        mapped_label = 'Very Negative'
                    else:
                        mapped_label = 'Negative'
                
                mapped_results.append({'label': mapped_label, 'score': score})
            
            return mapped_results
        
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return []
    
    def infer_article_topic(self, transcript_df):
        """Infer the article topic from the conversation content."""
        if transcript_df is None or transcript_df.empty:
            return "Unknown"
        
        # Get actual article URL if available
        article_url = transcript_df['article_url'].iloc[0] if not transcript_df['article_url'].iloc[0] == '' else None
        
        if article_url:
            return article_url
        
        # If no URL available, try to infer from content
        all_messages = ' '.join(transcript_df['message'].dropna().tolist())
        
        # Extract potential topics using keyword analysis
        sports_keywords = ['game', 'team', 'player', 'coach', 'score', 'season', 'match', 'championship', 'football', 'basketball', 'baseball', 'sports']
        politics_keywords = ['government', 'president', 'election', 'political', 'vote', 'congress', 'democrat', 'republican', 'policy', 'senate', 'law']
        business_keywords = ['company', 'market', 'stock', 'economy', 'business', 'finance', 'investment', 'trade', 'economic', 'industry', 'corporate']
        tech_keywords = ['technology', 'software', 'computer', 'app', 'internet', 'digital', 'tech', 'data', 'innovation', 'device', 'smartphone']
        
        # Count keyword occurrences
        topic_counts = {
            'Sports': sum(1 for keyword in sports_keywords if re.search(r'\b' + keyword + r'\b', all_messages, re.IGNORECASE)),
            'Politics': sum(1 for keyword in politics_keywords if re.search(r'\b' + keyword + r'\b', all_messages, re.IGNORECASE)),
            'Business': sum(1 for keyword in business_keywords if re.search(r'\b' + keyword + r'\b', all_messages, re.IGNORECASE)),
            'Technology': sum(1 for keyword in tech_keywords if re.search(r'\b' + keyword + r'\b', all_messages, re.IGNORECASE))
        }
        
        # Get topic with highest keyword count
        top_topic = max(topic_counts.items(), key=lambda x: x[1])
        
        if top_topic[1] > 0:
            return f"Likely a {top_topic[0]} article"
        else:
            return "Unknown article topic"
    
    def get_agent_message_counts(self, transcript_df):
        """Count messages by each agent."""
        if transcript_df is None or transcript_df.empty:
            return {}
        
        agent_counts = transcript_df['agent'].value_counts().to_dict()
        return agent_counts
    
    def analyze_agent_sentiment(self, transcript_df):
        """Analyze the overall sentiment for each agent."""
        if transcript_df is None or transcript_df.empty:
            return {}
        
        agent_sentiment = {}
        
        for agent in transcript_df['agent'].unique():
            agent_messages = transcript_df[transcript_df['agent'] == agent]['message'].tolist()
            
            # Use sentiment model to analyze
            sentiment_results = self.analyze_sentiment(agent_messages)
            
            if sentiment_results:
                # Count sentiment labels
                sentiment_counts = Counter([result['label'] for result in sentiment_results])
                
                # Calculate average sentiment score
                avg_score = sum([result['score'] for result in sentiment_results]) / len(sentiment_results)
                
                # Determine dominant sentiment
                dominant_sentiment = sentiment_counts.most_common(1)[0][0]
                
                # Calculate sentiment distribution percentages
                total = sum(sentiment_counts.values())
                sentiment_distribution = {sentiment: count / total * 100 for sentiment, count in sentiment_counts.items()}
                
                agent_sentiment[agent] = {
                    'dominant_sentiment': dominant_sentiment,
                    'avg_score': avg_score,
                    'sentiment_counts': dict(sentiment_counts),
                    'sentiment_distribution': sentiment_distribution
                }
            else:
                agent_sentiment[agent] = {'dominant_sentiment': 'Unknown', 'avg_score': 0.0}
        
        return agent_sentiment
    
    def analyze_transcript(self, transcript_data):
        """Analyze a transcript and return key insights."""
        # Load transcript
        transcript_df = self.load_transcript(transcript_data)
        
        if transcript_df is None or transcript_df.empty:
            return {
                'error': 'Failed to load or parse transcript data'
            }
        
        # Get article link/topic
        article_link = self.infer_article_topic(transcript_df)
        
        # Get message counts by agent
        message_counts = self.get_agent_message_counts(transcript_df)
        
        # Analyze sentiment by agent
        agent_sentiment = self.analyze_agent_sentiment(transcript_df)
        
        # Prepare results
        results = {
            'article_link': article_link,
            'message_counts': message_counts,
            'agent_sentiment': agent_sentiment,
            'conversation_id': transcript_df['conversation_id'].iloc[0] if not transcript_df.empty else 'unknown'
        }
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """Visualize analysis results."""
        if not results or 'error' in results:
            print("No valid results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Message counts
        agent_labels = list(results['message_counts'].keys())
        message_values = list(results['message_counts'].values())
        
        axes[0].bar(agent_labels, message_values, color=['#3498db', '#e74c3c'])
        axes[0].set_title('Message Count by Agent')
        axes[0].set_xlabel('Agent')
        axes[0].set_ylabel('Number of Messages')
        
        # Add count labels
        for i, count in enumerate(message_values):
            axes[0].text(i, count + 0.5, str(count), ha='center')
        
        # Sentiment analysis
        agent_sentiment = results['agent_sentiment']
        
        # Extract data for plotting
        agents = []
        sentiment_labels = []
        sentiment_values = []
        
        for agent, sentiment_data in agent_sentiment.items():
            for sentiment, percentage in sentiment_data.get('sentiment_distribution', {}).items():
                agents.append(agent)
                sentiment_labels.append(sentiment)
                sentiment_values.append(percentage)
        
        # Create DataFrame for plotting
        sentiment_df = pd.DataFrame({
            'Agent': agents,
            'Sentiment': sentiment_labels,
            'Percentage': sentiment_values
        })
        
        # Plot sentiment distribution as stacked bars
        if not sentiment_df.empty:
            sentiment_pivot = sentiment_df.pivot(index='Agent', columns='Sentiment', values='Percentage')
            sentiment_pivot.fillna(0, inplace=True)
            
            sentiment_pivot.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis')
            axes[1].set_title('Sentiment Distribution by Agent')
            axes[1].set_xlabel('Agent')
            axes[1].set_ylabel('Percentage (%)')
            axes[1].legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add article info
        fig.suptitle(f"Analysis of Conversation: {results['conversation_id']}\nArticle: {results['article_link']}", 
                     fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Results visualization saved to {save_path}")
        
        plt.close()
    
    def evaluate_model_accuracy(self, df, true_sentiment_col='sentiment', message_col='message', sample_size=100):
        """Evaluate the accuracy of the sentiment model against ground truth labels."""
        # Sample data for evaluation
        if len(df) > sample_size:
            eval_df = df.sample(sample_size, random_state=42)
        else:
            eval_df = df
        
        # Get predictions from model
        messages = eval_df[message_col].tolist()
        sentiment_results = self.analyze_sentiment(messages)
        
        if not sentiment_results:
            return {"error": "No sentiment predictions generated"}
        
        # Map ground truth labels to model's label space for comparison
        # This mapping depends on your dataset's specific sentiment labels
        sentiment_mapping = {
            'Neutral': 'Negative',  # Adjust based on your specific labels
            'Happy': 'Positive',
            'Positive': 'Positive',
            'Curious to dive deeper': 'Positive',
            'Surprised': 'Positive',
            'Sad': 'Negative',
            'Fearful': 'Negative',
            'Angry': 'Negative',
            'Disgusted': 'Negative',
            'Negative': 'Negative'
        }
        
        # Transform ground truth labels using mapping
        true_labels = eval_df[true_sentiment_col].map(lambda x: sentiment_mapping.get(x, 'Negative')).tolist()
        
        # Get predicted labels (simple positive/negative)
        pred_labels = [result['label'].split()[0] for result in sentiment_results]
        
        # Calculate accuracy and generate report
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels)
        
        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "sample_size": len(eval_df)
        }
        
        return results
    
    def run_accuracy_evaluation(self, data_path='processed_data.csv'):
        """Run accuracy evaluation on processed dataset."""
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            print(f"Loaded dataset for evaluation. Shape: {df.shape}")
            
            # Evaluate model accuracy
            eval_results = self.evaluate_model_accuracy(df)
            
            # Save results
            with open('transcript_analysis/model_evaluation.txt', 'w') as f:
                f.write("=== Sentiment Model Evaluation ===\n\n")
                f.write(f"Model: {self.sentiment_model_name}\n")
                f.write(f"Sample size: {eval_results.get('sample_size', 'N/A')}\n\n")
                f.write(f"Accuracy: {eval_results.get('accuracy', 'N/A'):.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(eval_results.get('classification_report', 'N/A'))
            
            print(f"Model evaluation saved to 'transcript_analysis/model_evaluation.txt'")
            return eval_results
            
        except Exception as e:
            print(f"Error in model evaluation: {e}")
            return {"error": str(e)}


# API function for transcript analysis
def analyze_chat_transcript(transcript_data):
    """
    Analyze a chat transcript and return key insights.
    
    Args:
        transcript_data: JSON data or file path to a transcript
    
    Returns:
        dict: Analysis results with the following information:
            - article_link: Possible article link or topic
            - message_counts: Number of messages by each agent
            - agent_sentiment: Overall sentiment analysis for each agent
    """
    analyzer = TranscriptAnalyzer()
    results = analyzer.analyze_transcript(transcript_data)
    
    # Save visualization
    if 'error' not in results:
        save_path = f"transcript_analysis/conversation_{results['conversation_id']}.png"
        analyzer.visualize_results(results, save_path)
    
    return results


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TranscriptAnalyzer()
    
    # Test with sample data or file
    try:
        # Load original dataset to extract a sample conversation
        with open("BiztelAI_DS_Dataset_Mar'25.json", 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Take first conversation as sample
        sample_convo_id = list(all_data.keys())[0]
        sample_convo = {sample_convo_id: all_data[sample_convo_id]}
        
        print(f"Analyzing sample conversation: {sample_convo_id}")
        
        # Analyze the sample conversation
        results = analyze_chat_transcript(sample_convo)
        
        # Print results
        print("\n=== Analysis Results ===")
        print(f"Article: {results['article_link']}")
        print("\nMessage Counts:")
        for agent, count in results['message_counts'].items():
            print(f"  {agent}: {count} messages")
        
        print("\nAgent Sentiment:")
        for agent, sentiment in results['agent_sentiment'].items():
            print(f"  {agent}: {sentiment['dominant_sentiment']} (score: {sentiment['avg_score']:.2f})")
            if 'sentiment_distribution' in sentiment:
                print("    Distribution:")
                for label, pct in sentiment['sentiment_distribution'].items():
                    print(f"      {label}: {pct:.1f}%")
        
        # Evaluate model accuracy
        print("\nEvaluating sentiment model accuracy...")
        analyzer.run_accuracy_evaluation()
        
    except Exception as e:
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
        print(f"Error in sample analysis: {e}") 