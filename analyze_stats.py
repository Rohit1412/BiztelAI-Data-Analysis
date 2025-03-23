import pandas as pd
import numpy as np

def load_data(file_path="processed_data.csv"):
    """Load the processed data."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def analyze_data(df):
    """Analyze the data and print statistical information."""
    if df is None:
        return
    
    # Basic statistics for numerical columns
    print("\n=== Numerical Columns Statistics ===")
    numerical_cols = ['message_length', 'word_count']
    for col in numerical_cols:
        if col in df.columns:
            stats = df[col].describe()
            print(f"\n{col} Statistics:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']}")
            print(f"  25%: {stats['25%']}")
            print(f"  50% (Median): {stats['50%']}")
            print(f"  75%: {stats['75%']}")
            print(f"  Max: {stats['max']}")
    
    # Categorical columns analysis
    print("\n=== Categorical Columns Analysis ===")
    categorical_cols = ['agent', 'config', 'sentiment', 'turn_rating']
    for col in categorical_cols:
        if col in df.columns:
            value_counts = df[col].value_counts()
            print(f"\n{col} Value Counts:")
            for value, count in value_counts.items():
                print(f"  {value}: {count} ({count/len(df)*100:.2f}%)")
    
    # Correlation between numerical columns
    print("\n=== Correlations ===")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Print correlations
        print("Correlation Matrix:")
        print(corr_matrix)
        
        # Print strongest correlations
        print("\nStrongest Correlations:")
        # Get upper triangle of correlation matrix (to avoid duplicates)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Stack correlation coefficients and sort
        strongest_corrs = upper.stack().sort_values(ascending=False)
        
        # Print top 10 or all if less than 10
        for i, ((col1, col2), corr) in enumerate(strongest_corrs.items()):
            print(f"  {col1} ⟷ {col2}: {corr:.4f}")
            if i >= 9:  # Print top 10
                break
    
    # Article distribution
    print("\n=== Article Analysis ===")
    article_counts = df['article_url'].value_counts()
    print(f"Number of unique articles: {len(article_counts)}")
    print("\nTop 5 most frequent articles:")
    for url, count in article_counts.head(5).items():
        print(f"  {url}: {count} messages")
    
    # Conversation analysis
    print("\n=== Conversation Analysis ===")
    conversation_counts = df['conversation_id'].value_counts()
    print(f"Number of unique conversations: {len(conversation_counts)}")
    
    # Messages per conversation statistics
    messages_per_conv = conversation_counts.describe()
    print("\nMessages per conversation statistics:")
    print(f"  Mean: {messages_per_conv['mean']:.2f}")
    print(f"  Min: {messages_per_conv['min']}")
    print(f"  Max: {messages_per_conv['max']}")
    
    # Agent interaction analysis
    print("\n=== Agent Interaction Analysis ===")
    # Group by conversation_id and agent, then count messages
    agent_conv_counts = df.groupby(['conversation_id', 'agent']).size().reset_index(name='message_count')
    
    # Calculate messages per agent in each conversation
    agent_message_stats = agent_conv_counts.groupby('agent')['message_count'].describe()
    
    print("\nMessages per conversation by agent:")
    print(agent_message_stats)

if __name__ == "__main__":
    df = load_data()
    analyze_data(df) 