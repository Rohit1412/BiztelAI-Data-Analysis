import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_processed_data(file_path='processed_data.csv'):
    """Load processed data from CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def plot_categorical_distributions(df):
    """Plot distributions of categorical variables."""
    categorical_cols = ['agent', 'config', 'turn_rating', 'sentiment']
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if col in df.columns:
            # Get top 10 categories if there are many
            value_counts = df[col].value_counts().head(10)
            
            # Plot
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            
            # Rotate x labels if needed
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('categorical_distributions.png')
    print("Categorical distributions plot saved as 'categorical_distributions.png'")

def plot_message_length_distribution(df):
    """Plot distribution of message lengths."""
    if 'message_length' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Remove outliers for better visualization
        q1 = df['message_length'].quantile(0.25)
        q3 = df['message_length'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        filtered_df = df[df['message_length'] <= upper_bound]
        
        # Plot histogram
        sns.histplot(filtered_df['message_length'], kde=True)
        plt.title('Distribution of Message Lengths')
        plt.xlabel('Message Length (characters)')
        plt.ylabel('Frequency')
        
        plt.savefig('message_length_distribution.png')
        print("Message length distribution plot saved as 'message_length_distribution.png'")

def plot_word_count_distribution(df):
    """Plot distribution of word counts."""
    if 'word_count' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Remove outliers for better visualization
        q1 = df['word_count'].quantile(0.25)
        q3 = df['word_count'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        filtered_df = df[df['word_count'] <= upper_bound]
        
        # Plot histogram
        sns.histplot(filtered_df['word_count'], kde=True)
        plt.title('Distribution of Word Counts')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.savefig('word_count_distribution.png')
        print("Word count distribution plot saved as 'word_count_distribution.png'")

def plot_sentiment_by_agent(df):
    """Plot sentiment distribution by agent."""
    if 'agent' in df.columns and 'sentiment' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a crosstab
        cross_tab = pd.crosstab(df['agent'], df['sentiment'])
        cross_tab_normalized = cross_tab.div(cross_tab.sum(axis=1), axis=0)
        
        # Plot
        cross_tab_normalized.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Sentiment Distribution by Agent')
        plt.xlabel('Agent')
        plt.ylabel('Proportion')
        plt.legend(title='Sentiment')
        
        plt.tight_layout()
        plt.savefig('sentiment_by_agent.png')
        print("Sentiment by agent plot saved as 'sentiment_by_agent.png'")

def plot_turn_rating_distribution(df):
    """Plot distribution of turn ratings."""
    if 'turn_rating' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Count values
        value_counts = df['turn_rating'].value_counts().sort_index()
        
        # Plot
        ax = value_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Turn Ratings')
        plt.xlabel('Turn Rating')
        plt.ylabel('Count')
        
        # Add count labels on bars
        for i, count in enumerate(value_counts):
            ax.text(i, count + 5, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig('turn_rating_distribution.png')
        print("Turn rating distribution plot saved as 'turn_rating_distribution.png'")

def main():
    """Main function to generate all visualizations."""
    # Load data
    df = load_processed_data()
    
    if df is not None:
        # Create output directory if it doesn't exist
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
        
        # Generate visualizations
        plot_categorical_distributions(df)
        plot_message_length_distribution(df)
        plot_word_count_distribution(df)
        plot_sentiment_by_agent(df)
        plot_turn_rating_distribution(df)
        
        print("All visualizations created successfully!")

if __name__ == "__main__":
    main() 