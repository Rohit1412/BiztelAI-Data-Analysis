<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ExploratoryDataAnalysis:
    """Class for conducting exploratory data analysis on the processed dataset."""
    
    def __init__(self, data_path='processed_data.csv'):
        """Initialize with the path to the processed data."""
        self.data_path = data_path
        self.df = None
        self.load_data()
        
        # Create output directory for visualizations
        if not os.path.exists('eda_visualizations'):
            os.makedirs('eda_visualizations')
    
    def load_data(self):
        """Load the processed data."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def generate_summary_statistics(self):
        """Generate and print summary statistics."""
        if self.df is None:
            return
        
        print("\n=== Dataset Summary ===")
        print(f"Total messages: {len(self.df)}")
        print(f"Unique conversations: {self.df['conversation_id'].nunique()}")
        print(f"Unique articles: {self.df['article_url'].nunique()}")
        
        # Article-level summary
        article_summary = self.df.groupby('article_url').agg(
            message_count=('message', 'count'),
            conversation_count=('conversation_id', 'nunique'),
            avg_message_length=('message_length', 'mean'),
            avg_word_count=('word_count', 'mean')
        ).sort_values('message_count', ascending=False)
        
        print("\n=== Article-Level Summary (Top 5) ===")
        print(article_summary.head())
        
        # Agent-level summary
        agent_summary = self.df.groupby('agent').agg(
            message_count=('message', 'count'),
            avg_message_length=('message_length', 'mean'),
            avg_word_count=('word_count', 'mean')
        )
        
        print("\n=== Agent-Level Summary ===")
        print(agent_summary)
        
        # Sentiment distribution by agent
        sentiment_by_agent = pd.crosstab(self.df['agent'], self.df['sentiment'])
        sentiment_by_agent_pct = sentiment_by_agent.div(sentiment_by_agent.sum(axis=1), axis=0) * 100
        
        print("\n=== Sentiment Distribution by Agent (%) ===")
        print(sentiment_by_agent_pct.round(2))
        
        # Save summary stats to file
        with open('eda_visualizations/summary_statistics.txt', 'w') as f:
            f.write("=== Dataset Summary ===\n")
            f.write(f"Total messages: {len(self.df)}\n")
            f.write(f"Unique conversations: {self.df['conversation_id'].nunique()}\n")
            f.write(f"Unique articles: {self.df['article_url'].nunique()}\n\n")
            
            f.write("=== Article-Level Summary (Top 10) ===\n")
            f.write(article_summary.head(10).to_string())
            f.write("\n\n")
            
            f.write("=== Agent-Level Summary ===\n")
            f.write(agent_summary.to_string())
            f.write("\n\n")
            
            f.write("=== Sentiment Distribution by Agent (%) ===\n")
            f.write(sentiment_by_agent_pct.round(2).to_string())
        
        print("\nSummary statistics saved to 'eda_visualizations/summary_statistics.txt'")
        
        return {
            'article_summary': article_summary,
            'agent_summary': agent_summary,
            'sentiment_by_agent': sentiment_by_agent_pct
        }
    
    def analyze_conversation_structure(self):
        """Analyze the structure of conversations."""
        if self.df is None:
            return
        
        # Messages per conversation
        messages_per_conv = self.df.groupby('conversation_id')['message'].count()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(messages_per_conv, kde=True)
        plt.title('Distribution of Messages per Conversation')
        plt.xlabel('Number of Messages')
        plt.ylabel('Frequency')
        plt.savefig('eda_visualizations/messages_per_conversation.png')
        plt.close()
        
        # Messages by agent per conversation
        messages_by_agent = self.df.groupby(['conversation_id', 'agent'])['message'].count().unstack()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=messages_by_agent['agent_1'] if 'agent_1' in messages_by_agent.columns else [],
            y=messages_by_agent['agent_2'] if 'agent_2' in messages_by_agent.columns else []
        )
        plt.title('Messages by Agent 1 vs Agent 2 per Conversation')
        plt.xlabel('Agent 1 Messages')
        plt.ylabel('Agent 2 Messages')
        plt.savefig('eda_visualizations/agent_message_comparison.png')
        plt.close()
        
        # Turn-taking patterns
        self.df['prev_agent'] = self.df.groupby('conversation_id')['agent'].shift(1)
        turn_taking = pd.crosstab(self.df['prev_agent'], self.df['agent'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(turn_taking, annot=True, cmap='Blues', fmt='d')
        plt.title('Turn-Taking Patterns')
        plt.xlabel('Current Agent')
        plt.ylabel('Previous Agent')
        plt.savefig('eda_visualizations/turn_taking_patterns.png')
        plt.close()
        
        print("\nConversation structure analysis saved to 'eda_visualizations/'")
    
    def analyze_message_content(self):
        """Analyze message content and text characteristics."""
        if self.df is None:
            return
        
        # Word count vs message sentiment
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='sentiment', y='word_count', data=self.df)
        plt.title('Word Count by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Word Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('eda_visualizations/word_count_by_sentiment.png')
        plt.close()
        
        # Most common words by agent
        stop_words = set(stopwords.words('english'))
        
        for agent in self.df['agent'].unique():
            agent_messages = self.df[self.df['agent'] == agent]
            
            # Extract all words from lemmatized tokens
            all_words = []
            for tokens in agent_messages['lemmatized'].dropna():
                if isinstance(tokens, list):
                    all_words.extend([word.lower() for word in tokens if word.lower() not in stop_words])
                elif isinstance(tokens, str):
                    # Handle case where lemmatized might be a string representation of a list
                    try:
                        word_list = eval(tokens)
                        if isinstance(word_list, list):
                            all_words.extend([word.lower() for word in word_list if word.lower() not in stop_words])
                    except:
                        pass
            
            # Create word frequency dictionary
            word_freq = Counter(all_words)
            
            # Generate word cloud
            if word_freq:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Most Common Words for {agent}')
                plt.tight_layout()
                plt.savefig(f'eda_visualizations/wordcloud_{agent}.png')
                plt.close()
                
                # Bar chart of top words
                top_words = dict(word_freq.most_common(20))
                
                plt.figure(figsize=(12, 6))
                plt.bar(top_words.keys(), top_words.values())
                plt.title(f'Top 20 Words for {agent}')
                plt.xlabel('Word')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'eda_visualizations/top_words_{agent}.png')
                plt.close()
        
        print("\nMessage content analysis saved to 'eda_visualizations/'")
    
    def analyze_sentiment_patterns(self):
        """Analyze sentiment patterns in the dataset."""
        if self.df is None:
            return
        
        # Sentiment distribution
        plt.figure(figsize=(12, 6))
        sentiment_counts = self.df['sentiment'].value_counts()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_distribution.png')
        plt.close()
        
        # Sentiment by agent
        plt.figure(figsize=(12, 8))
        agent_sentiment = pd.crosstab(self.df['agent'], self.df['sentiment'])
        agent_sentiment_pct = agent_sentiment.div(agent_sentiment.sum(axis=1), axis=0)
        
        ax = agent_sentiment_pct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
        plt.title('Sentiment Distribution by Agent')
        plt.xlabel('Agent')
        plt.ylabel('Proportion')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_by_agent.png')
        plt.close()
        
        # Sentiment by conversation position
        self.df['position_in_conv'] = self.df.groupby('conversation_id').cumcount() + 1
        max_position = min(20, self.df['position_in_conv'].max())  # Limit to first 20 positions or max available
        
        position_sentiment = pd.crosstab(
            self.df[self.df['position_in_conv'] <= max_position]['position_in_conv'],
            self.df[self.df['position_in_conv'] <= max_position]['sentiment']
        )
        
        position_sentiment_pct = position_sentiment.div(position_sentiment.sum(axis=1), axis=0)
        
        plt.figure(figsize=(14, 8))
        ax = position_sentiment_pct.plot(kind='line', marker='o')
        plt.title('Sentiment Evolution Throughout Conversations')
        plt.xlabel('Message Position in Conversation')
        plt.ylabel('Proportion')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_by_position.png')
        plt.close()
        
        print("\nSentiment pattern analysis saved to 'eda_visualizations/'")
    
    def analyze_article_patterns(self):
        """Analyze patterns related to articles."""
        if self.df is None:
            return
        
        # Top 10 articles by message count
        top_articles = self.df['article_url'].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_articles)), top_articles.values)
        plt.xticks(range(len(top_articles)), [f"Article {i+1}" for i in range(len(top_articles))], rotation=45)
        plt.title('Top 10 Articles by Message Count')
        plt.xlabel('Article')
        plt.ylabel('Message Count')
        plt.tight_layout()
        plt.savefig('eda_visualizations/top_articles.png')
        plt.close()
        
        # Extract article types from URLs
        def extract_article_type(url):
            if 'sports' in url:
                return 'Sports'
            elif 'politics' in url:
                return 'Politics'
            elif 'business' in url or 'economic' in url:
                return 'Business'
            elif 'health' in url:
                return 'Health'
            elif 'technology' in url or 'tech' in url:
                return 'Technology'
            elif 'entertainment' in url:
                return 'Entertainment'
            else:
                return 'Other'
        
        self.df['article_type'] = self.df['article_url'].apply(extract_article_type)
        
        # Article type distribution
        article_type_counts = self.df['article_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        plt.pie(article_type_counts.values, labels=article_type_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Article Type Distribution')
        plt.tight_layout()
        plt.savefig('eda_visualizations/article_type_distribution.png')
        plt.close()
        
        # Sentiment by article type
        article_type_sentiment = pd.crosstab(self.df['article_type'], self.df['sentiment'])
        article_type_sentiment_pct = article_type_sentiment.div(article_type_sentiment.sum(axis=1), axis=0)
        
        plt.figure(figsize=(14, 8))
        article_type_sentiment_pct.plot(kind='bar', stacked=True)
        plt.title('Sentiment Distribution by Article Type')
        plt.xlabel('Article Type')
        plt.ylabel('Proportion')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_by_article_type.png')
        plt.close()
        
        print("\nArticle pattern analysis saved to 'eda_visualizations/'")
    
    def analyze_turn_ratings(self):
        """Analyze turn ratings in the dataset."""
        if self.df is None or 'turn_rating' not in self.df.columns:
            return
        
        # Turn rating distribution
        plt.figure(figsize=(10, 6))
        turn_rating_counts = self.df['turn_rating'].value_counts().sort_index()
        sns.barplot(x=turn_rating_counts.index, y=turn_rating_counts.values)
        plt.title('Turn Rating Distribution')
        plt.xlabel('Turn Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('eda_visualizations/turn_rating_distribution.png')
        plt.close()
        
        # Turn rating by agent
        turn_rating_by_agent = pd.crosstab(self.df['agent'], self.df['turn_rating'])
        turn_rating_by_agent_pct = turn_rating_by_agent.div(turn_rating_by_agent.sum(axis=1), axis=0)
        
        plt.figure(figsize=(12, 6))
        turn_rating_by_agent_pct.plot(kind='bar', stacked=True)
        plt.title('Turn Rating Distribution by Agent')
        plt.xlabel('Agent')
        plt.ylabel('Proportion')
        plt.legend(title='Turn Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/turn_rating_by_agent.png')
        plt.close()
        
        # Turn rating vs sentiment
        turn_rating_sentiment = pd.crosstab(self.df['turn_rating'], self.df['sentiment'])
        turn_rating_sentiment_pct = turn_rating_sentiment.div(turn_rating_sentiment.sum(axis=1), axis=0)
        
        plt.figure(figsize=(14, 8))
        turn_rating_sentiment_pct.plot(kind='bar', stacked=True)
        plt.title('Sentiment Distribution by Turn Rating')
        plt.xlabel('Turn Rating')
        plt.ylabel('Proportion')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_by_turn_rating.png')
        plt.close()
        
        # Turn rating vs message length
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='turn_rating', y='message_length', data=self.df)
        plt.title('Message Length by Turn Rating')
        plt.xlabel('Turn Rating')
        plt.ylabel('Message Length (characters)')
        plt.tight_layout()
        plt.savefig('eda_visualizations/message_length_by_turn_rating.png')
        plt.close()
        
        print("\nTurn rating analysis saved to 'eda_visualizations/'")
    
    def generate_interactive_visuals(self):
        """Generate interactive visualizations using Plotly."""
        if self.df is None:
            return
        
        # Prepare data
        agent_stats = self.df.groupby('agent').agg({
            'message': 'count',
            'message_length': 'mean',
            'word_count': 'mean'
        }).reset_index()
        
        # Interactive bar chart - Message count by agent
        fig1 = px.bar(
            agent_stats, 
            x='agent', 
            y='message',
            color='agent',
            title='Message Count by Agent',
            labels={'message': 'Number of Messages', 'agent': 'Agent'},
            template='plotly_white'
        )
        fig1.write_html('eda_visualizations/interactive_message_count.html')
        
        # Interactive scatter plot - Message length vs Word count by Agent
        fig2 = px.scatter(
            self.df, 
            x='message_length', 
            y='word_count',
            color='agent',
            hover_data=['sentiment'],
            title='Message Length vs Word Count by Agent',
            labels={
                'message_length': 'Message Length (characters)',
                'word_count': 'Word Count',
                'agent': 'Agent',
                'sentiment': 'Sentiment'
            },
            template='plotly_white'
        )
        fig2.write_html('eda_visualizations/interactive_message_metrics.html')
        
        # Interactive heatmap - Sentiment by Agent
        sentiment_agent = pd.crosstab(self.df['sentiment'], self.df['agent'])
        
        fig3 = px.imshow(
            sentiment_agent,
            labels=dict(x="Agent", y="Sentiment", color="Count"),
            title="Sentiment by Agent Heatmap",
            template='plotly_white'
        )
        fig3.write_html('eda_visualizations/interactive_sentiment_heatmap.html')
        
        # Interactive sunburst chart - Article Type -> Config -> Sentiment
        fig4 = px.sunburst(
            self.df,
            path=['article_type', 'config', 'sentiment'],
            title='Hierarchical View: Article Type -> Config -> Sentiment',
            template='plotly_white'
        )
        fig4.write_html('eda_visualizations/interactive_hierarchy.html')
        
        print("\nInteractive visualizations saved to 'eda_visualizations/'")
    
    def run_complete_eda(self):
        """Run all EDA functions."""
        print("Running complete exploratory data analysis...")
        self.generate_summary_statistics()
        self.analyze_conversation_structure()
        self.analyze_message_content()
        self.analyze_sentiment_patterns()
        self.analyze_article_patterns()
        self.analyze_turn_ratings()
        self.generate_interactive_visuals()
        print("\nExploratory data analysis completed. All visualizations saved to 'eda_visualizations/'")


if __name__ == "__main__":
    eda = ExploratoryDataAnalysis()
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ExploratoryDataAnalysis:
    """Class for conducting exploratory data analysis on the processed dataset."""
    
    def __init__(self, data_path='processed_data.csv'):
        """Initialize with the path to the processed data."""
        self.data_path = data_path
        self.df = None
        self.load_data()
        
        # Create output directory for visualizations
        if not os.path.exists('eda_visualizations'):
            os.makedirs('eda_visualizations')
    
    def load_data(self):
        """Load the processed data."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def generate_summary_statistics(self):
        """Generate and print summary statistics."""
        if self.df is None:
            return
        
        print("\n=== Dataset Summary ===")
        print(f"Total messages: {len(self.df)}")
        print(f"Unique conversations: {self.df['conversation_id'].nunique()}")
        print(f"Unique articles: {self.df['article_url'].nunique()}")
        
        # Article-level summary
        article_summary = self.df.groupby('article_url').agg(
            message_count=('message', 'count'),
            conversation_count=('conversation_id', 'nunique'),
            avg_message_length=('message_length', 'mean'),
            avg_word_count=('word_count', 'mean')
        ).sort_values('message_count', ascending=False)
        
        print("\n=== Article-Level Summary (Top 5) ===")
        print(article_summary.head())
        
        # Agent-level summary
        agent_summary = self.df.groupby('agent').agg(
            message_count=('message', 'count'),
            avg_message_length=('message_length', 'mean'),
            avg_word_count=('word_count', 'mean')
        )
        
        print("\n=== Agent-Level Summary ===")
        print(agent_summary)
        
        # Sentiment distribution by agent
        sentiment_by_agent = pd.crosstab(self.df['agent'], self.df['sentiment'])
        sentiment_by_agent_pct = sentiment_by_agent.div(sentiment_by_agent.sum(axis=1), axis=0) * 100
        
        print("\n=== Sentiment Distribution by Agent (%) ===")
        print(sentiment_by_agent_pct.round(2))
        
        # Save summary stats to file
        with open('eda_visualizations/summary_statistics.txt', 'w') as f:
            f.write("=== Dataset Summary ===\n")
            f.write(f"Total messages: {len(self.df)}\n")
            f.write(f"Unique conversations: {self.df['conversation_id'].nunique()}\n")
            f.write(f"Unique articles: {self.df['article_url'].nunique()}\n\n")
            
            f.write("=== Article-Level Summary (Top 10) ===\n")
            f.write(article_summary.head(10).to_string())
            f.write("\n\n")
            
            f.write("=== Agent-Level Summary ===\n")
            f.write(agent_summary.to_string())
            f.write("\n\n")
            
            f.write("=== Sentiment Distribution by Agent (%) ===\n")
            f.write(sentiment_by_agent_pct.round(2).to_string())
        
        print("\nSummary statistics saved to 'eda_visualizations/summary_statistics.txt'")
        
        return {
            'article_summary': article_summary,
            'agent_summary': agent_summary,
            'sentiment_by_agent': sentiment_by_agent_pct
        }
    
    def analyze_conversation_structure(self):
        """Analyze the structure of conversations."""
        if self.df is None:
            return
        
        # Messages per conversation
        messages_per_conv = self.df.groupby('conversation_id')['message'].count()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(messages_per_conv, kde=True)
        plt.title('Distribution of Messages per Conversation')
        plt.xlabel('Number of Messages')
        plt.ylabel('Frequency')
        plt.savefig('eda_visualizations/messages_per_conversation.png')
        plt.close()
        
        # Messages by agent per conversation
        messages_by_agent = self.df.groupby(['conversation_id', 'agent'])['message'].count().unstack()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=messages_by_agent['agent_1'] if 'agent_1' in messages_by_agent.columns else [],
            y=messages_by_agent['agent_2'] if 'agent_2' in messages_by_agent.columns else []
        )
        plt.title('Messages by Agent 1 vs Agent 2 per Conversation')
        plt.xlabel('Agent 1 Messages')
        plt.ylabel('Agent 2 Messages')
        plt.savefig('eda_visualizations/agent_message_comparison.png')
        plt.close()
        
        # Turn-taking patterns
        self.df['prev_agent'] = self.df.groupby('conversation_id')['agent'].shift(1)
        turn_taking = pd.crosstab(self.df['prev_agent'], self.df['agent'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(turn_taking, annot=True, cmap='Blues', fmt='d')
        plt.title('Turn-Taking Patterns')
        plt.xlabel('Current Agent')
        plt.ylabel('Previous Agent')
        plt.savefig('eda_visualizations/turn_taking_patterns.png')
        plt.close()
        
        print("\nConversation structure analysis saved to 'eda_visualizations/'")
    
    def analyze_message_content(self):
        """Analyze message content and text characteristics."""
        if self.df is None:
            return
        
        # Word count vs message sentiment
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='sentiment', y='word_count', data=self.df)
        plt.title('Word Count by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Word Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('eda_visualizations/word_count_by_sentiment.png')
        plt.close()
        
        # Most common words by agent
        stop_words = set(stopwords.words('english'))
        
        for agent in self.df['agent'].unique():
            agent_messages = self.df[self.df['agent'] == agent]
            
            # Extract all words from lemmatized tokens
            all_words = []
            for tokens in agent_messages['lemmatized'].dropna():
                if isinstance(tokens, list):
                    all_words.extend([word.lower() for word in tokens if word.lower() not in stop_words])
                elif isinstance(tokens, str):
                    # Handle case where lemmatized might be a string representation of a list
                    try:
                        word_list = eval(tokens)
                        if isinstance(word_list, list):
                            all_words.extend([word.lower() for word in word_list if word.lower() not in stop_words])
                    except:
                        pass
            
            # Create word frequency dictionary
            word_freq = Counter(all_words)
            
            # Generate word cloud
            if word_freq:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Most Common Words for {agent}')
                plt.tight_layout()
                plt.savefig(f'eda_visualizations/wordcloud_{agent}.png')
                plt.close()
                
                # Bar chart of top words
                top_words = dict(word_freq.most_common(20))
                
                plt.figure(figsize=(12, 6))
                plt.bar(top_words.keys(), top_words.values())
                plt.title(f'Top 20 Words for {agent}')
                plt.xlabel('Word')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'eda_visualizations/top_words_{agent}.png')
                plt.close()
        
        print("\nMessage content analysis saved to 'eda_visualizations/'")
    
    def analyze_sentiment_patterns(self):
        """Analyze sentiment patterns in the dataset."""
        if self.df is None:
            return
        
        # Sentiment distribution
        plt.figure(figsize=(12, 6))
        sentiment_counts = self.df['sentiment'].value_counts()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_distribution.png')
        plt.close()
        
        # Sentiment by agent
        plt.figure(figsize=(12, 8))
        agent_sentiment = pd.crosstab(self.df['agent'], self.df['sentiment'])
        agent_sentiment_pct = agent_sentiment.div(agent_sentiment.sum(axis=1), axis=0)
        
        ax = agent_sentiment_pct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
        plt.title('Sentiment Distribution by Agent')
        plt.xlabel('Agent')
        plt.ylabel('Proportion')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_by_agent.png')
        plt.close()
        
        # Sentiment by conversation position
        self.df['position_in_conv'] = self.df.groupby('conversation_id').cumcount() + 1
        max_position = min(20, self.df['position_in_conv'].max())  # Limit to first 20 positions or max available
        
        position_sentiment = pd.crosstab(
            self.df[self.df['position_in_conv'] <= max_position]['position_in_conv'],
            self.df[self.df['position_in_conv'] <= max_position]['sentiment']
        )
        
        position_sentiment_pct = position_sentiment.div(position_sentiment.sum(axis=1), axis=0)
        
        plt.figure(figsize=(14, 8))
        ax = position_sentiment_pct.plot(kind='line', marker='o')
        plt.title('Sentiment Evolution Throughout Conversations')
        plt.xlabel('Message Position in Conversation')
        plt.ylabel('Proportion')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_by_position.png')
        plt.close()
        
        print("\nSentiment pattern analysis saved to 'eda_visualizations/'")
    
    def analyze_article_patterns(self):
        """Analyze patterns related to articles."""
        if self.df is None:
            return
        
        # Top 10 articles by message count
        top_articles = self.df['article_url'].value_counts().head(10)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_articles)), top_articles.values)
        plt.xticks(range(len(top_articles)), [f"Article {i+1}" for i in range(len(top_articles))], rotation=45)
        plt.title('Top 10 Articles by Message Count')
        plt.xlabel('Article')
        plt.ylabel('Message Count')
        plt.tight_layout()
        plt.savefig('eda_visualizations/top_articles.png')
        plt.close()
        
        # Extract article types from URLs
        def extract_article_type(url):
            if 'sports' in url:
                return 'Sports'
            elif 'politics' in url:
                return 'Politics'
            elif 'business' in url or 'economic' in url:
                return 'Business'
            elif 'health' in url:
                return 'Health'
            elif 'technology' in url or 'tech' in url:
                return 'Technology'
            elif 'entertainment' in url:
                return 'Entertainment'
            else:
                return 'Other'
        
        self.df['article_type'] = self.df['article_url'].apply(extract_article_type)
        
        # Article type distribution
        article_type_counts = self.df['article_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        plt.pie(article_type_counts.values, labels=article_type_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Article Type Distribution')
        plt.tight_layout()
        plt.savefig('eda_visualizations/article_type_distribution.png')
        plt.close()
        
        # Sentiment by article type
        article_type_sentiment = pd.crosstab(self.df['article_type'], self.df['sentiment'])
        article_type_sentiment_pct = article_type_sentiment.div(article_type_sentiment.sum(axis=1), axis=0)
        
        plt.figure(figsize=(14, 8))
        article_type_sentiment_pct.plot(kind='bar', stacked=True)
        plt.title('Sentiment Distribution by Article Type')
        plt.xlabel('Article Type')
        plt.ylabel('Proportion')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_by_article_type.png')
        plt.close()
        
        print("\nArticle pattern analysis saved to 'eda_visualizations/'")
    
    def analyze_turn_ratings(self):
        """Analyze turn ratings in the dataset."""
        if self.df is None or 'turn_rating' not in self.df.columns:
            return
        
        # Turn rating distribution
        plt.figure(figsize=(10, 6))
        turn_rating_counts = self.df['turn_rating'].value_counts().sort_index()
        sns.barplot(x=turn_rating_counts.index, y=turn_rating_counts.values)
        plt.title('Turn Rating Distribution')
        plt.xlabel('Turn Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('eda_visualizations/turn_rating_distribution.png')
        plt.close()
        
        # Turn rating by agent
        turn_rating_by_agent = pd.crosstab(self.df['agent'], self.df['turn_rating'])
        turn_rating_by_agent_pct = turn_rating_by_agent.div(turn_rating_by_agent.sum(axis=1), axis=0)
        
        plt.figure(figsize=(12, 6))
        turn_rating_by_agent_pct.plot(kind='bar', stacked=True)
        plt.title('Turn Rating Distribution by Agent')
        plt.xlabel('Agent')
        plt.ylabel('Proportion')
        plt.legend(title='Turn Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/turn_rating_by_agent.png')
        plt.close()
        
        # Turn rating vs sentiment
        turn_rating_sentiment = pd.crosstab(self.df['turn_rating'], self.df['sentiment'])
        turn_rating_sentiment_pct = turn_rating_sentiment.div(turn_rating_sentiment.sum(axis=1), axis=0)
        
        plt.figure(figsize=(14, 8))
        turn_rating_sentiment_pct.plot(kind='bar', stacked=True)
        plt.title('Sentiment Distribution by Turn Rating')
        plt.xlabel('Turn Rating')
        plt.ylabel('Proportion')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('eda_visualizations/sentiment_by_turn_rating.png')
        plt.close()
        
        # Turn rating vs message length
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='turn_rating', y='message_length', data=self.df)
        plt.title('Message Length by Turn Rating')
        plt.xlabel('Turn Rating')
        plt.ylabel('Message Length (characters)')
        plt.tight_layout()
        plt.savefig('eda_visualizations/message_length_by_turn_rating.png')
        plt.close()
        
        print("\nTurn rating analysis saved to 'eda_visualizations/'")
    
    def generate_interactive_visuals(self):
        """Generate interactive visualizations using Plotly."""
        if self.df is None:
            return
        
        # Prepare data
        agent_stats = self.df.groupby('agent').agg({
            'message': 'count',
            'message_length': 'mean',
            'word_count': 'mean'
        }).reset_index()
        
        # Interactive bar chart - Message count by agent
        fig1 = px.bar(
            agent_stats, 
            x='agent', 
            y='message',
            color='agent',
            title='Message Count by Agent',
            labels={'message': 'Number of Messages', 'agent': 'Agent'},
            template='plotly_white'
        )
        fig1.write_html('eda_visualizations/interactive_message_count.html')
        
        # Interactive scatter plot - Message length vs Word count by Agent
        fig2 = px.scatter(
            self.df, 
            x='message_length', 
            y='word_count',
            color='agent',
            hover_data=['sentiment'],
            title='Message Length vs Word Count by Agent',
            labels={
                'message_length': 'Message Length (characters)',
                'word_count': 'Word Count',
                'agent': 'Agent',
                'sentiment': 'Sentiment'
            },
            template='plotly_white'
        )
        fig2.write_html('eda_visualizations/interactive_message_metrics.html')
        
        # Interactive heatmap - Sentiment by Agent
        sentiment_agent = pd.crosstab(self.df['sentiment'], self.df['agent'])
        
        fig3 = px.imshow(
            sentiment_agent,
            labels=dict(x="Agent", y="Sentiment", color="Count"),
            title="Sentiment by Agent Heatmap",
            template='plotly_white'
        )
        fig3.write_html('eda_visualizations/interactive_sentiment_heatmap.html')
        
        # Interactive sunburst chart - Article Type -> Config -> Sentiment
        fig4 = px.sunburst(
            self.df,
            path=['article_type', 'config', 'sentiment'],
            title='Hierarchical View: Article Type -> Config -> Sentiment',
            template='plotly_white'
        )
        fig4.write_html('eda_visualizations/interactive_hierarchy.html')
        
        print("\nInteractive visualizations saved to 'eda_visualizations/'")
    
    def run_complete_eda(self):
        """Run all EDA functions."""
        print("Running complete exploratory data analysis...")
        self.generate_summary_statistics()
        self.analyze_conversation_structure()
        self.analyze_message_content()
        self.analyze_sentiment_patterns()
        self.analyze_article_patterns()
        self.analyze_turn_ratings()
        self.generate_interactive_visuals()
        print("\nExploratory data analysis completed. All visualizations saved to 'eda_visualizations/'")


if __name__ == "__main__":
    eda = ExploratoryDataAnalysis()
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    eda.run_complete_eda() 